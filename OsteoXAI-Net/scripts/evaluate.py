import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from osteoxai.experiments import save_classification_plots
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--out',default='results/figures');a=p.parse_args()
print(json.dumps(save_classification_plots(pd.read_csv(a.predictions),a.out,a.threshold),indent=2))
