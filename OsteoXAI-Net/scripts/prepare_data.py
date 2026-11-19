import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from osteoxai.data import inventory,fixed_splits,deduplicate
p=argparse.ArgumentParser();p.add_argument('--mendeley',required=True);p.add_argument('--kaggle');p.add_argument('--out',default='data/manifests');p.add_argument('--deduplicate',action='store_true');p.add_argument('--no-pretrained',action='store_true');a=p.parse_args()
o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
primary=inventory(a.mendeley,o/'mendeley_all.csv');print('Primary:',primary.groupby('class_name').size().to_dict())
fixed_splits(primary).to_csv(o/'fixed_split.csv',index=False)
if a.kaggle:
    ext=inventory(a.kaggle,o/'kaggle_all.csv');print('Kaggle:',ext.groupby('class_name').size().to_dict())
    if a.deduplicate:
        retained,audit=deduplicate(primary[primary.valid],ext[ext.valid],o/'dedup_audit.csv',pretrained=not a.no_pretrained)
        retained.to_csv(o/'kaggle_deduplicated.csv',index=False);print('Retained:',len(retained),'flagged:',audit.suspected_duplicate.sum())
