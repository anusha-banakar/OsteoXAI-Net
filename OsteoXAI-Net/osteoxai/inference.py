
import numpy as np
import torch
from torch.nn import functional as F
from .model import set_mc_dropout

def predict_mc(model,batch,T=30,device='cpu'):
    x=batch['image'].to(device);w=batch['wavelets'].to(device)
    set_mc_dropout(model,enabled=T>1)
    with torch.no_grad():
        probs=torch.stack([model(x,w).sigmoid().view(-1) for _ in range(T)],dim=0)
    model.eval()
    return probs.mean(0).cpu().numpy(),probs.var(0,unbiased=False).cpu().numpy()

def fit_uncertainty_stats(variances):
    a,b=np.quantile(np.asarray(variances),[.05,.95]);return {'u05':float(a),'u95':float(b)}

def reliability(p,v,stats):
    p=np.asarray(p);v=np.asarray(v)
    u=np.clip((v-stats['u05'])/(max(stats['u95']-stats['u05'],1e-8)),0,1)
    conf=np.maximum(p,1-p)
    return u,conf,conf*(1-u)

def normalized_map(x):
    x=np.asarray(x,dtype=np.float64);x=x-np.nanmin(x);return np.nan_to_num(x/max(np.nanmax(x),1e-8)).astype(np.float32)

def gradcam_plus(model,image,wavelets,positive_class=True):
    """Grad-CAM++ with genuine higher-order gradients on the final dense stage.

    Uses log-probability exponentiation of a class score for higher derivatives,
    including the stable commonly used derivative-coefficient formulation.
    """
    model.eval()
    act=[];grad=[]
    def forward_hook(m,inp,out):
        act.append(out)
        out.register_hook(lambda g:grad.append(g))
    target=model.spatial.backbone.norm5
    h=target.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            out=model(image,wavelets)
            score=(out if positive_class else -out).sum()
            # exp(logit) produces nonzero higher-order gradients.
            score=torch.exp(score.clamp(max=15))
            g=torch.autograd.grad(score,act[0],retain_graph=False)[0]
            a=act[0].detach();g=g.detach()
            g2=g*g;g3=g2*g
            alpha=g2/(2*g2+(a*g3).sum((2,3),keepdim=True)+1e-8)
            alpha=torch.where(torch.isfinite(alpha),alpha,torch.zeros_like(alpha))
            weights=(alpha*F.relu(g)).sum((2,3),keepdim=True)
            cam=F.relu((weights*a).sum(1,keepdim=True))
            cam=F.interpolate(cam,size=image.shape[-2:],mode='bilinear',align_corners=False)
            return np.stack([normalized_map(z[0]) for z in cam.detach().cpu().numpy()])
    finally:h.remove()

def explain(model,batch,device='cpu',positive=None):
    model.eval();x=batch['image'].to(device);w=batch['wavelets'].to(device)
    with torch.no_grad():
        outputs=model(x,w,return_maps=True);pr=outputs['logits'].sigmoid().flatten().cpu().numpy()
        att=F.interpolate(outputs['spatial_map'],size=x.shape[-2:],mode='bilinear',align_corners=False).cpu().numpy()
    if positive is None:positive=pr>=.5
    # Individual backprop prevents cross-example gradients and allows per-class target selection.
    cams=[]
    for i in range(len(pr)):
        cams.append(gradcam_plus(model,x[i:i+1],w[i:i+1],bool(positive[i]))[0])
    sa=np.stack([normalized_map(z[0]) for z in att]);cams=np.stack(cams)
    cross=outputs['cross_attention']
    return {'gradcam_pp':cams,'spatial_attention':sa,'cross_attention':cross.detach().cpu().numpy() if cross is not None else None,'spectral_importance':outputs['spectral_importance'].detach().cpu().numpy() if outputs['spectral_importance'] is not None else None}

def explanation_consistency(cams,spatial,saliency_threshold=.65,mix=.5):
    out=[]
    for a,b in zip(cams,spatial):
        a=np.asarray(a).ravel();b=np.asarray(b).ravel()
        rho=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-9 and np.std(b)>1e-9 else 0.
        if not np.isfinite(rho):rho=0.
        ma=a>=saliency_threshold;mb=b>=saliency_threshold
        union=np.logical_or(ma,mb).sum();iou=float(np.logical_and(ma,mb).sum()/union) if union else 1.
        out.append({'correlation':rho,'iou':iou,'consistency':float(mix*max(0,rho)+(1-mix)*iou)})
    return out

def trust_decision(p,v,consistency,stats,config):
    u,conf,rel=reliability(p,v,stats)
    c=np.asarray(consistency);weight=config.get('trust_weight',.5)
    trust=weight*rel+(1-weight)*c
    accepted=(u<=config['uncertainty_threshold']) & (rel>=config['reliability_threshold']) & (c>=config['consistency_threshold']) & (trust>=config['trust_threshold'])
    return [{'confidence':float(conf[i]),'normalized_uncertainty':float(u[i]),'reliability':float(rel[i]),'consistency':float(c[i]),'trust':float(trust[i]),'accepted':bool(accepted[i])} for i in range(len(p))]

def fit_trust_thresholds(p,v,consistency,y=None,stats=None,target_coverage=.8):
    """Validation-only quantile thresholds; a target coverage is specified a priori."""
    if stats is None:stats=fit_uncertainty_stats(v)
    u,conf,rel=reliability(p,v,stats);c=np.asarray(consistency);s=.5*rel+.5*c
    q=max(0,min(1,1-target_coverage))
    return {'uncertainty_threshold':float(np.quantile(u,1-q)), 'reliability_threshold':float(np.quantile(rel,q)), 'consistency_threshold':float(np.quantile(c,q)), 'trust_threshold':float(np.quantile(s,q)), 'trust_weight':.5,'target_coverage':target_coverage}
