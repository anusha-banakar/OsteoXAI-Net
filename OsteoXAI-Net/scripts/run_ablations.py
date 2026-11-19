import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np
from osteoxai.data import fixed_splits
from osteoxai.train import train_one
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',default='configs/default.json');p.add_argument('--out',default='results/ablations');a=p.parse_args()
df=pd.read_csv(a.manifest);df=df[df.label.isin([0,1]) & df.valid.astype(bool)].reset_index(drop=True)
if 'split' not in df:df=fixed_splits(df)
ids=[np.where(df.split==s)[0] for s in ('train','val','test')];cfg=json.loads(Path(a.config).read_text())
results=[]
for name in ['backbone','bone','spectral','transformer','attention','full']:
    conf={**cfg,'ablation':name};r=train_one(df,*ids,conf,Path(a.out)/name);results.append({'configuration':name,**r})
Path(a.out).mkdir(parents=True,exist_ok=True);pd.DataFrame(results).to_csv(Path(a.out)/'summary.csv',index=False)
