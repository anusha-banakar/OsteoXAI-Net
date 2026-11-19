"""Logistic regression clinical-variable baseline trained on the same manifest split.

Clinical fields are loaded from a user-authored CSV keyed by image_id. No synthetic
clinical covariates are invented if the source dataset does not expose them.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from osteoxai.metrics import classification_metrics,select_threshold
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--clinical-csv',required=True);p.add_argument('--out',default='results/clinical');a=p.parse_args()
df=pd.read_csv(a.manifest).merge(pd.read_csv(a.clinical_csv),on='image_id',validate='one_to_one');df=df[df.label.isin([0,1])]
if 'split' not in df:raise ValueError('Manifest must have train/val/test split labels.')
cols=[c for c in df.columns if c not in ('path','image_id','split','valid','label','class_name')]
if not cols:raise ValueError('No clinical columns were supplied.')
num=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])];cat=[c for c in cols if c not in num]
trans=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')),cat)])
pipe=make_pipeline(trans,LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))
train=df[df.split=='train'];val=df[df.split=='val'];test=df[df.split=='test'];pipe.fit(train[cols],train.label)
threshold=select_threshold(val.label,pipe.predict_proba(val[cols])[:,1]);prob=pipe.predict_proba(test[cols])[:,1]
res=classification_metrics(test.label,prob,threshold);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
(out/'metrics.json').write_text(json.dumps(res,indent=2));pd.DataFrame({'image_id':test.image_id,'label':test.label,'probability':prob}).to_csv(out/'predictions.csv',index=False)
print(json.dumps(res,indent=2))
