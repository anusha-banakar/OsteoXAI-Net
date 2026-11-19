import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd,cv2
from osteoxai.train import create_model,loader
from osteoxai.data import wavelet_bands
from osteoxai.experiments import perturb
from osteoxai.metrics import classification_metrics
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',default='results/robustness');a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];st=ck['preprocessing'];device='cuda' if torch.cuda.is_available() else 'cpu';model=create_model({**cfg,'pretrained':False}).to(device);model.load_state_dict(ck['model']);model.eval()
df=pd.read_csv(a.manifest);df=df[df.label.isin([0,1]) & df.valid.astype(bool)];rows=[]
for kind in ['clean','noise','blur','contrast','brightness','rotation','occlusion','compression']:
    ps=[];ys=[]
    for b in loader(df,st,False,{'batch_size':1},42):
        im=b['image'][0] if kind=='clean' else perturb(b['image'][0],kind)
        # Recompute wavelets after perturbation: both pathways see the same modified radiograph.
        bands=wavelet_bands(im[0].numpy())
        w=np.stack([cv2.resize((x-m)/max(s,1e-5),(16,16),interpolation=cv2.INTER_AREA) for x,m,s in zip(bands,st['wavelet_mean'],st['wavelet_std'])])
        with torch.no_grad():prob=model(im[None].to(device),torch.tensor(w[None],dtype=torch.float32,device=device)).sigmoid().item()
        ps.append(prob);ys.append(int(b['label'][0]))
    rows.append({'perturbation':kind,**classification_metrics(ys,ps,cfg.get('threshold',.5))})
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'robustness.csv',index=False)
