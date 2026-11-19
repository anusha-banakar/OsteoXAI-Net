"""Baseline models, perturbation robustness, calibration plots and report generation."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd,torch,cv2
from torch import nn
from torchvision import models
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score
from .metrics import classification_metrics,bootstrap_ci,risk_coverage,decision_curve

class ImageBaseline(nn.Module):
    def __init__(self,name='densenet121',pretrained=True):
        super().__init__();self.name=name
        if name=='densenet121':
            m=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None);dim=m.classifier.in_features;m.classifier=nn.Linear(dim,1)
        elif name=='resnet50':
            m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None);dim=m.fc.in_features;m.fc=nn.Linear(dim,1)
        elif name=='efficientnet_b0':
            m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None);dim=m.classifier[-1].in_features;m.classifier[-1]=nn.Linear(dim,1)
        elif name=='vit_b_16':
            m=models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None);dim=m.heads.head.in_features;m.heads.head=nn.Linear(dim,1)
        else:raise ValueError(f'Unknown baseline {name}')
        self.net=m
    def forward(self,image,wavelets=None):return self.net(image.repeat(1,3,1,1))

def save_classification_plots(df,out,threshold=.5):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    y=df.label.to_numpy();p=df.probability.to_numpy()
    if len(np.unique(y))==2:
        fpr,tpr,_=roc_curve(y,p);fig,ax=plt.subplots(figsize=(5,4));ax.plot(fpr,tpr,label=f'ROC-AUC = {roc_auc_score(y,p):.3f}');ax.plot([0,1],[0,1],ls='--');ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate');ax.legend();fig.tight_layout();fig.savefig(out/'roc.png',dpi=600);plt.close(fig)
        pr,re,_=precision_recall_curve(y,p);fig,ax=plt.subplots(figsize=(5,4));ax.plot(re,pr);ax.set(xlabel='Recall',ylabel='Precision');fig.tight_layout();fig.savefig(out/'pr.png',dpi=600);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));ax.hist(p[y==0],bins=12,alpha=.5,label='Normal');ax.hist(p[y==1],bins=12,alpha=.5,label='Osteoporosis');ax.legend();ax.set(xlabel='Osteoporosis Probability',ylabel='Count');fig.tight_layout();fig.savefig(out/'probabilities.png',dpi=600);plt.close(fig)
    stats=classification_metrics(y,p,threshold)
    stats['auc_bootstrap_95ci']=bootstrap_ci(y,p,lambda a,b:roc_auc_score(a,b),n_boot=2000) if len(np.unique(y))==2 else [None,None]
    (out/'metrics.json').write_text(json.dumps(stats,indent=2));pd.DataFrame(decision_curve(y,p)).to_csv(out/'decision_curve.csv',index=False)
    return stats

def perturb(img,kind,severity=1):
    """img is a single standardized torch [1,H,W] tensor."""
    a=img.detach().cpu().numpy()[0]; h,w=a.shape
    if kind=='noise':z=a+np.random.default_rng(42).normal(0,.05*severity,a.shape)
    elif kind=='blur':z=cv2.GaussianBlur(a,(2*severity+1,2*severity+1),0)
    elif kind=='contrast':z=(a-a.mean())*(1-.15*severity)+a.mean()
    elif kind=='brightness':z=a+.12*severity
    elif kind=='rotation':
        mat=cv2.getRotationMatrix2D((w/2,h/2),3*severity,1);z=cv2.warpAffine(a,mat,(w,h),borderMode=cv2.BORDER_REFLECT_101)
    elif kind=='occlusion':
        z=a.copy();hh=max(1,int(h*.1*severity));ww=max(1,int(w*.1*severity));z[h//2-hh//2:h//2+hh//2,w//2-ww//2:w//2+ww//2]=a.mean()
    elif kind=='compression':
        norm=cv2.normalize(a,None,0,255,cv2.NORM_MINMAX).astype('uint8');ok,enc=cv2.imencode('.jpg',norm,[cv2.IMWRITE_JPEG_QUALITY,max(10,100-20*severity)]);z=cv2.imdecode(enc,cv2.IMREAD_GRAYSCALE).astype('float32')/255*(a.max()-a.min())+a.min()
    else:raise ValueError(kind)
    return torch.tensor(z[None].copy(),dtype=img.dtype)

# Baseline training protocol reuses the same batch, weighted-BCE and checkpoint rules.
def train_baseline(df,tr_idx,va_idx,te_idx,cfg,out,model_name='densenet121',seed=42):
    from .train import seed_everything,loader,batch_predict
    from .data import fit_preprocessing
    from .metrics import select_threshold
    out=Path(out);out.mkdir(parents=True,exist_ok=True);seed_everything(seed)
    tr,va,te=[df.iloc[i] for i in (tr_idx,va_idx,te_idx)]
    stats=fit_preprocessing(tr.path,cfg.get('image_size',224))
    device=cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu')
    model=ImageBaseline(model_name,cfg.get('pretrained',True)).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=cfg.get('learning_rate',1e-4),weight_decay=cfg.get('weight_decay',1e-4))
    counts=tr.label.value_counts();weights=torch.tensor([len(tr)/(2*counts.get(j,1)) for j in (0,1)],device=device,dtype=torch.float32)
    best=-1;beststate=None;wait=0;history=[]
    for epoch in range(cfg.get('epochs',100)):
        model.train();losses=[]
        for b in loader(tr,stats,True,cfg,seed+epoch):
            x=b['image'].to(device);y=b['label'].to(device);optim.zero_grad(set_to_none=True)
            z=model(x).flatten();l=(nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')*torch.where(y>.5,weights[1],weights[0])).mean()
            l.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();losses.append(l.item())
        y,p,_=batch_predict(model,loader(va,stats,False,cfg,seed),device)
        score=roc_auc_score(y,p) if len(np.unique(y))==2 else .5
        history.append({'epoch':epoch+1,'loss':np.mean(losses),'val_auc':score})
        if score>best+cfg.get('min_delta',1e-4):best=score;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0
        else:wait+=1
        if wait>=cfg.get('patience',15):break
    model.load_state_dict(beststate)
    yv,pv,_=batch_predict(model,loader(va,stats,False,cfg,seed),device);threshold=select_threshold(yv,pv)
    y,p,ids=batch_predict(model,loader(te,stats,False,cfg,seed),device)
    metrics=classification_metrics(y,p,threshold)
    pd.DataFrame({'image_id':ids,'label':y,'probability':p}).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    torch.save({'model':model.state_dict(),'baseline':model_name,'preprocessing':stats,'threshold':threshold},out/'best.pt')
    return metrics
