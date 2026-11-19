"""Complete train/validation/evaluation with optional Optuna and repeated stratified CV."""
import os, json, random, time, math, copy
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,f1_score
from .data import fit_preprocessing,RadiographDataset,split_inner,repeated_folds
from .model import OsteoXAINet,attention_entropy_penalty
from .metrics import classification_metrics,select_threshold
from .inference import predict_mc,fit_uncertainty_stats

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def create_model(cfg):
    return OsteoXAINet(pretrained=cfg.get('pretrained',True),d=cfg.get('d',256),layers=cfg.get('layers',4),heads=cfg.get('heads',8),n_queries=cfg.get('queries',8),cross_heads=cfg.get('cross_heads',8),ablation=cfg.get('ablation','full'))

def loader(df,stats,training,cfg,seed):
    dataset=RadiographDataset(df,stats,training=training,seed=seed)
    return DataLoader(dataset,batch_size=cfg.get('batch_size',16),shuffle=training,num_workers=cfg.get('workers',0),pin_memory=torch.cuda.is_available())

def batch_predict(model,ld,device):
    model.eval();ys=[];ps=[];ident=[]
    with torch.no_grad():
        for b in ld:
            p=model(b['image'].to(device),b['wavelets'].to(device)).sigmoid().view(-1).cpu().numpy()
            ys.extend(b['label'].numpy());ps.extend(p);ident.extend(b['image_id'])
    return np.asarray(ys),np.asarray(ps),ident

def train_one(df,train_idx,val_idx,test_idx,cfg,out,seed=42):
    seed_everything(seed);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tr=df.iloc[train_idx].copy();va=df.iloc[val_idx].copy();te=df.iloc[test_idx].copy()
    stats=fit_preprocessing(tr.path,cfg.get('image_size',224));(out/'preprocess.json').write_text(json.dumps(stats,indent=2))
    pd.concat([tr.assign(split='train'),va.assign(split='val'),te.assign(split='test')]).to_csv(out/'split.csv',index=False)
    dl_tr=loader(tr,stats,True,cfg,seed);dl_va=loader(va,stats,False,cfg,seed);dl_te=loader(te,stats,False,cfg,seed)
    device=torch.device(cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    model=create_model(cfg).to(device)
    pretrained_params=list(model.spatial.backbone.parameters());pretrained_ids={id(p) for p in pretrained_params}
    other=[p for p in model.parameters() if id(p) not in pretrained_ids]
    lr=cfg.get('learning_rate',1e-4)
    opt=torch.optim.AdamW([{'params':pretrained_params,'lr':lr*.1,'initial_lr':lr*.1},{'params':other,'lr':lr,'initial_lr':lr}],weight_decay=cfg.get('weight_decay',1e-4))
    max_epochs=cfg.get('epochs',100);warmup=cfg.get('warmup',5);min_lr=cfg.get('min_lr',1e-6)
    counts=tr.label.value_counts();class_weights=torch.tensor([len(tr)/(2*counts.get(j,1)) for j in [0,1]],device=device,dtype=torch.float32)
    best=-np.inf;best_state=None;stale=0;history=[]
    for epoch in range(max_epochs):
        if epoch<warmup:factor=(epoch+1)/max(1,warmup)
        else:factor=(min_lr/lr)+(1-min_lr/lr)*.5*(1+math.cos(math.pi*(epoch-warmup)/max(1,max_epochs-warmup-1)))
        for group in opt.param_groups:group['lr']=group['initial_lr']*factor
        # Frozen lower backbone stage only during warmup; higher features remain trainable.
        for name,p in model.spatial.backbone.named_parameters():p.requires_grad= not(epoch<warmup and any(z in name for z in ['conv0','norm0','denseblock1','transition1']))
        model.train();dl_tr.dataset.epoch=epoch;losses=[]
        for b in dl_tr:
            x=b['image'].to(device);wave=b['wavelets'].to(device);y=b['label'].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=cfg.get('amp',True) and device.type=='cuda'):
                z=model(x,wave,return_maps=True)
                logits=z['logits'].flatten();weights=torch.where(y>0.5,class_weights[1],class_weights[0])
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).mean()
                penalty=attention_entropy_penalty(z['cross_attention'],cfg.get('entropy_margin',1.4))
                if penalty is not None:loss=loss+cfg.get('attention_lambda',.01)*penalty
                if cfg.get('explicit_l2',0)>0:loss=loss+cfg['explicit_l2']*sum((p*p).sum() for p in model.parameters() if p.requires_grad)
            if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at epoch {epoch}')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.detach().cpu()))
        vy,vp,_=batch_predict(model,dl_va,device)
        auc=roc_auc_score(vy,vp) if len(np.unique(vy))==2 else 0.5
        f1=f1_score(vy,vp>=.5,zero_division=0)
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_auc':float(auc),'val_f1':float(f1),'lr':opt.param_groups[-1]['lr']})
        if auc>best+cfg.get('min_delta',1e-4):best=auc;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=cfg.get('patience',15):break
    model.load_state_dict(best_state);model.eval()
    vy,vp,_=batch_predict(model,dl_va,device);threshold=select_threshold(vy,vp)
    y,p,ids=batch_predict(model,dl_te,device)
    # Fit MC variance quantiles using validation only; inference scoring is separate.
    vs=[]
    for b in dl_va:
        _,v=predict_mc(model,b,T=cfg.get('mc_passes',30),device=device);vs.extend(v)
    u_stats=fit_uncertainty_stats(vs)
    result=classification_metrics(y,p,threshold)
    pd.DataFrame({'image_id':ids,'label':y.astype(int),'probability':p,'predicted':(p>=threshold).astype(int)}).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    config=dict(cfg);config.update({'threshold':threshold,'uncertainty':u_stats,'best_val_auc':float(best),'seed':seed})
    (out/'model_config.json').write_text(json.dumps(config,indent=2))
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    torch.save({'model':model.state_dict(),'config':config,'preprocessing':stats},out/'best.pt')
    return result

def run_cv(df,cfg,out,repeats=5,folds=5):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    all_results=[];all_predictions=[]
    seeds=[42,123,2024,3407,7777]
    for rep,fold,tridx,teidx in repeated_folds(df,repeats,folds,seeds):
        tr,va=split_inner(df,tridx,seed=seeds[rep%len(seeds)]+fold,n_val=cfg.get('inner_val',10))
        path=out/f'repeat{rep+1}_fold{fold+1}'
        result=train_one(df,tr,va,teidx,cfg,path,seed=seeds[rep%len(seeds)]+fold)
        all_results.append({'repeat':rep,'fold':fold,**result})
        tmp=pd.read_csv(path/'predictions.csv');tmp['repeat']=rep;tmp['fold']=fold;all_predictions.append(tmp)
    pd.DataFrame(all_results).to_csv(out/'fold_metrics.csv',index=False)
    predictions=pd.concat(all_predictions,ignore_index=True);predictions.to_csv(out/'oof_predictions.csv',index=False)
    # Primary estimates: score 85 OOF predictions per repetition, then mean ± sample SD over repeats.
    reps=[]
    for rep,group in predictions.groupby('repeat'):
        vals=classification_metrics(group.label,group.probability,threshold=.5);reps.append({'repeat':int(rep),**vals})
    df_rep=pd.DataFrame(reps);df_rep.to_csv(out/'repeat_metrics.csv',index=False)
    summary={k:{'mean':float(df_rep[k].mean()),'sd':float(df_rep[k].std(ddof=1)) if len(df_rep)>1 else 0.} for k in ['accuracy','f1','roc_auc','mcc','ece','brier']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary
