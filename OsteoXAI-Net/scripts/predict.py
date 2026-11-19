import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2,torch
from osteoxai.data import read_gray,enhance,wavelet_bands
from osteoxai.train import create_model
from osteoxai.inference import predict_mc,explain,explanation_consistency,trust_decision
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--out',default='results/prediction');p.add_argument('--explain',action='store_true');p.add_argument('--mc-passes',type=int,default=30);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);device='cuda' if torch.cuda.is_available() else 'cpu'
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];st=ck['preprocessing']
model=create_model({**cfg,'pretrained':False}).to(device);model.load_state_dict(ck['model']);model.eval()
raw=read_gray(a.image,st['size']);prep=enhance(raw,st)
img=(prep.astype(np.float32)/255-st['mean'])/st['std'];wv=wavelet_bands(img)
wv=np.stack([cv2.resize((b-m)/max(s,1e-5),(16,16),interpolation=cv2.INTER_AREA) for b,m,s in zip(wv,st['wavelet_mean'],st['wavelet_std'])])
b={'image':torch.from_numpy(img[None,None]).float(),'wavelets':torch.from_numpy(wv[None].astype('float32'))}
pmean,variance=predict_mc(model,b,T=a.mc_passes,device=device)
r={'predicted_class':'Osteoporosis' if pmean[0]>=cfg.get('threshold',.5) else 'Normal','probability':float(pmean[0]),'variance':float(variance[0]),'threshold':cfg.get('threshold',.5),'interpretation':'QUS-referenced risk screening research only; not a clinical diagnosis'}
if a.explain:
    ex=explain(model,b,device,positive=np.asarray([pmean[0]>=cfg.get('threshold',.5)]))
    cs=explanation_consistency(ex['gradcam_pp'],ex['spatial_attention']);r['explanation_consistency']=cs[0]
    for key in ['gradcam_pp','spatial_attention']:
        heat=np.uint8(ex[key][0]*255);heat=cv2.applyColorMap(heat,cv2.COLORMAP_JET)
        overlay=cv2.addWeighted(cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR),.6,heat,.4,0);cv2.imwrite(str(out/f'{key}.png'),overlay)
    if ex['cross_attention'] is not None:np.save(out/'cross_attention.npy',ex['cross_attention'])
    if ex['spectral_importance'] is not None:np.save(out/'spectral_importance.npy',ex['spectral_importance'])
    if 'trust_thresholds' in cfg:
        r['trust']=trust_decision(pmean,variance,[cs[0]['consistency']],cfg['uncertainty'],cfg['trust_thresholds'])[0]
(out/'prediction.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
