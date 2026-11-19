import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np
from osteoxai.data import fixed_splits
from osteoxai.experiments import train_baseline
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',default='configs/default.json');p.add_argument('--out',default='results/baselines');p.add_argument('--models',nargs='+',default=['densenet121','resnet50','efficientnet_b0','vit_b_16']);a=p.parse_args()
df=pd.read_csv(a.manifest);df=df[df.label.isin([0,1]) & df.valid.astype(bool)].reset_index(drop=True)
if 'split' not in df:df=fixed_splits(df)
indices=[np.where(df.split==s)[0] for s in ('train','val','test')];cfg=json.loads(Path(a.config).read_text())
for name in a.models:print(name,train_baseline(df,*indices,cfg,Path(a.out)/name,model_name=name))
