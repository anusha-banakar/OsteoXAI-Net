import numpy as np,torch
from osteoxai.data import wavelet_bands
from osteoxai.model import OsteoXAINet,set_mc_dropout
from osteoxai.inference import explain,predict_mc,explanation_consistency,reliability
from osteoxai.metrics import classification_metrics,bootstrap_ci

def lightweight():return OsteoXAINet(pretrained=False,d=64,layers=1,heads=4,n_queries=4,cross_heads=4)

def test_wavelets():
    bands=wavelet_bands(np.random.default_rng(1).normal(size=(64,64)).astype('float32'))
    assert len(bands)==10 and all(np.isfinite(b).all() for b in bands)
    assert bands[0].shape[0]<bands[-1].shape[0]

def test_shape_backward():
    torch.set_num_threads(2);m=lightweight().train();x=torch.randn(2,1,64,64);w=torch.randn(2,10,16,16)
    z=m(x,w,return_maps=True);assert z['logits'].shape==(2,1)
    assert z['cross_attention'].shape==(2,4,4,10)
    assert z['spectral_importance'].shape==(2,10)
    z['logits'].sum().backward()
    assert m.cross.attn.in_proj_weight.grad.abs().sum()>0
    assert m.spectral.project[0].weight.grad.abs().sum()>0

def test_mc_explanations():
    torch.set_num_threads(2);m=lightweight();b={'image':torch.randn(1,1,64,64),'wavelets':torch.randn(1,10,16,16)}
    p,v=predict_mc(m,b,3);assert 0<=p[0]<=1 and v[0]>=0
    x=explain(m,b);assert x['gradcam_pp'].shape==(1,64,64)
    assert x['spatial_attention'].shape==(1,64,64)
    c=explanation_consistency(x['gradcam_pp'],x['spatial_attention']);assert 0<=c[0]['consistency']<=1

def test_metrics():
    y=np.array([0,1,0,1]);p=np.array([.1,.9,.4,.8]);r=classification_metrics(y,p)
    assert r['accuracy']==1 and r['roc_auc']==1
    assert len(bootstrap_ci(y,p,lambda a,b:classification_metrics(a,b)['accuracy'],n_boot=10))==2
    u,c,rel=reliability(p,np.zeros_like(p),{'u05':0,'u95':1})
    assert np.all((rel>=0)&(rel<=1))
