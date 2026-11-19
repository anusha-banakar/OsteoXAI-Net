"""Validation-only Optuna search; each trial retrains on precisely the same split."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np,optuna
from osteoxai.data import fixed_splits
from osteoxai.train import train_one
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',default='configs/default.json');p.add_argument('--trials',type=int,default=50);p.add_argument('--out',default='results/tuning');a=p.parse_args()
cfg=json.loads(Path(a.config).read_text());df=pd.read_csv(a.manifest);df=df[df.label.isin([0,1]) & df.valid.astype(bool)].reset_index(drop=True)
if 'split' not in df:df=fixed_splits(df)
tr=np.where(df.split=='train')[0];va=np.where(df.split=='val')[0];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
# train_one evaluates a provided third group, thus we pass validation as third to score trials; it does NOT touch fixed test.
def objective(trial):
    conf={**cfg,'layers':trial.suggest_categorical('layers',[2,4]),'heads':trial.suggest_categorical('heads',[4,8]),'learning_rate':trial.suggest_float('learning_rate',3e-5,3e-4,log=True),'attention_lambda':trial.suggest_float('attention_lambda',1e-4,.05,log=True),'weight_decay':trial.suggest_float('weight_decay',1e-6,1e-3,log=True)}
    # Inner validation used for early stopping; outer fixed validation used for ranking.
    from osteoxai.data import split_inner
    inner_tr,inner_va=split_inner(df,tr,seed=42+trial.number,n_val=max(4,min(10,len(tr)//5)))
    metrics=train_one(df,inner_tr,inner_va,va,conf,out/f'trial_{trial.number}',42+trial.number)
    trial.set_user_attr('metrics',metrics)
    return .7*(metrics['roc_auc'] or 0)+.3*metrics['f1']
study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42));study.optimize(objective,n_trials=a.trials)
(out/'best.json').write_text(json.dumps({'parameters':study.best_params,'score':study.best_value},indent=2));study.trials_dataframe().to_csv(out/'trials.csv',index=False)
