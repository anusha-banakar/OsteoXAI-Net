import argparse,sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,torch,numpy as np
from osteoxai.train import create_model,loader
from osteoxai.inference import (explain,predict_mc,explanation_consistency,fit_uncertainty_stats,fit_trust_thresholds)
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',default='checkpoints/calibrated.pt');p.add_argument('--coverage',type=float,default=.8);p.add_argument('--mc-passes',type=int,default=30);a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=copy.deepcopy(ck['config']);st=ck['preprocessing'];df=pd.read_csv(a.manifest)
if 'split' not in df:raise ValueError('Calibration manifest MUST contain a split column explicitly identifying validation cases.')
df=df[df.split.eq('val') & df.label.isin([0,1]) & df.valid.astype(bool)]
if df.empty:raise ValueError('No validation rows available for calibration.')
device='cuda' if torch.cuda.is_available() else 'cpu';model=create_model({**cfg,'pretrained':False}).to(device);model.load_state_dict(ck['model']);model.eval()
probs=[];variances=[];consistency=[]
for batch in loader(df,st,False,{'batch_size':1},42):
    p,v=predict_mc(model,batch,T=a.mc_passes,device=device)
    ex=explain(model,batch,device,positive=np.array([p[0]>=cfg.get('threshold',.5)]))
    cs=explanation_consistency(ex['gradcam_pp'],ex['spatial_attention'])[0]['consistency']
    probs.extend(p);variances.extend(v);consistency.append(cs)
ust=fit_uncertainty_stats(variances);thresholds=fit_trust_thresholds(probs,variances,consistency,stats=ust,target_coverage=a.coverage)
cfg['uncertainty']=ust;cfg['trust_thresholds']=thresholds;cfg['trust_calibration_n']=len(df)
ck['config']=cfg;out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,out)
print(json.dumps({'calibrated_checkpoint':str(out),'validation_images':len(df),'uncertainty':ust,'trust_thresholds':thresholds},indent=2))
