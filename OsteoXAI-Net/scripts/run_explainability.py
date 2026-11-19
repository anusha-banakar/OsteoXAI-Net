import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np,torch,cv2
from osteoxai.train import create_model,loader
from osteoxai.inference import explain,explanation_consistency,predict_mc,fit_trust_thresholds,fit_uncertainty_stats,trust_decision
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',default='results/explainability');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];st=ck['preprocessing'];device='cuda' if torch.cuda.is_available() else 'cpu'
model=create_model({**cfg,'pretrained':False}).to(device);model.load_state_dict(ck['model']);model.eval()
df=pd.read_csv(a.manifest);df=df[df.label.isin([0,1]) & df.valid.astype(bool)]
if a.limit:df=df.head(a.limit)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
rows=[]
for batch in loader(df,st,False,{'batch_size':1},42):
    p,v=predict_mc(model,batch,T=cfg.get('mc_passes',30),device=device)
    ex=explain(model,batch,device,positive=p>=cfg.get('threshold',.5))
    c=explanation_consistency(ex['gradcam_pp'],ex['spatial_attention'])[0]
    name=str(batch['image_id'][0]);rows.append({'image_id':name,'label':int(batch['label'][0]),'probability':float(p[0]),'variance':float(v[0]),**c})
    for key in ('gradcam_pp','spatial_attention'):
        im=cv2.applyColorMap(np.uint8(ex[key][0]*255),cv2.COLORMAP_JET);cv2.imwrite(str(out/f'{name}_{key}.png'),im)
    if ex['cross_attention'] is not None:np.save(out/f'{name}_cross_attention.npy',ex['cross_attention'])
results=pd.DataFrame(rows)
if len(results):
    stats=ck['config'].get('uncertainty',fit_uncertainty_stats(results.variance))
    thresholds=cfg.get('trust_thresholds')
    if thresholds is not None:
        decisions=trust_decision(results.probability.to_numpy(),results.variance.to_numpy(),results.consistency.to_numpy(),stats,thresholds)
        for key in decisions[0]:results[key]=[v[key] for v in decisions]
    else:
        print('Trust thresholds not calibrated: run scripts/calibrate_trust.py on a validation-only manifest first.')
results.to_csv(out/'explanations.csv',index=False)
