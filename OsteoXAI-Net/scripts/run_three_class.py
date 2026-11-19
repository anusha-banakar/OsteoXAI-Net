"""Three-class extension: uses full model, replaces binary logit head and CE objective."""
import argparse,sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from torch import nn
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_auc_score
from osteoxai.train import create_model,loader,seed_everything
from osteoxai.data import fit_preprocessing
p=argparse.ArgumentParser();p.add_argument('--train-manifest',required=True);p.add_argument('--test-manifest',required=True);p.add_argument('--config',default='configs/default.json');p.add_argument('--out',default='results/three_class');a=p.parse_args()
seed_everything(42);cfg=json.loads(Path(a.config).read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
df=pd.read_csv(a.train_manifest);df=df[df.label.isin([0,1,2]) & df.valid.astype(bool)].reset_index(drop=True)
te=pd.read_csv(a.test_manifest);te=te[te.label.isin([0,1,2]) & te.valid.astype(bool)].reset_index(drop=True)
# preserve class mapping 0 Normal, 1 Osteoporosis, 2 Osteopenia; report accordingly.
from sklearn.model_selection import train_test_split
ids=np.arange(len(df));tr,va=train_test_split(ids,test_size=.2,random_state=42,stratify=df.label)
st=fit_preprocessing(df.iloc[tr].path,cfg.get('image_size',224));model=create_model({**cfg,'pretrained':cfg.get('pretrained',True)}).to(device)
model.head[-1]=nn.Linear(128,3).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg.get('learning_rate',1e-4));best=-1;beststate=None;wait=0
counts=df.iloc[tr].label.value_counts();weights=torch.tensor([len(tr)/(3*counts.get(i,1)) for i in range(3)],dtype=torch.float32,device=device)
def evaluate(ids):
    model.eval();y=[];pr=[]
    with torch.no_grad():
        for b in loader(ids,st,False,cfg,42):
            p=model(b['image'].to(device),b['wavelets'].to(device)).softmax(1).cpu().numpy();pr.extend(p);y.extend(b['label'].numpy().astype(int))
    return np.asarray(y),np.asarray(pr)
for epoch in range(cfg.get('epochs',100)):
    model.train()
    for b in loader(df.iloc[tr],st,True,cfg,42+epoch):
        opt.zero_grad();z=model(b['image'].to(device),b['wavelets'].to(device));loss=nn.functional.cross_entropy(z,b['label'].long().to(device),weight=weights);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    y,pr=evaluate(df.iloc[va]);score=f1_score(y,pr.argmax(1),average='macro')
    if score>best:best=score;beststate=copy.deepcopy(model.state_dict());wait=0
    else:wait+=1
    if wait>=cfg.get('patience',15):break
model.load_state_dict(beststate);y,pr=evaluate(te);pred=pr.argmax(1)
res={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),'macro_ovr_roc_auc':float(roc_auc_score(y,pr,multi_class='ovr',average='macro'))}
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'metrics.json').write_text(json.dumps(res,indent=2));pd.DataFrame({'path':te.path,'label':y,'predicted':pred,'p_normal':pr[:,0],'p_osteoporosis':pr[:,1],'p_osteopenia':pr[:,2]}).to_csv(out/'predictions.csv',index=False);torch.save({'model':model.state_dict(),'preprocessing':st,'config':cfg,'num_classes':3},out/'best.pt');print(json.dumps(res,indent=2))
