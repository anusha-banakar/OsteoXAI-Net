import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import densenet121,DenseNet121_Weights

class ChannelSpatialAttention(nn.Module):
    def __init__(self,c=256,reduction=16):
        super().__init__();self.mlp=nn.Sequential(nn.Conv2d(c,c//reduction,1),nn.ReLU(inplace=True),nn.Conv2d(c//reduction,c,1))
        self.spatial=nn.Conv2d(2,1,7,padding=3)
        self.scale=nn.Parameter(torch.tensor(.1))
    def forward(self,x):
        ca=torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x,1))+self.mlp(F.adaptive_max_pool2d(x,1)))
        z=x*ca;sa=torch.sigmoid(self.spatial(torch.cat([z.mean(1,keepdim=True),z.amax(1,keepdim=True)],1)))
        return x+self.scale*z*sa,sa

class SpatialEncoder(nn.Module):
    def __init__(self,d=256,pretrained=True,grid=7):
        super().__init__()
        self.backbone=densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None).features
        self.proj=nn.ModuleList([nn.Sequential(nn.Conv2d(c,d,1,bias=False),nn.BatchNorm2d(d),nn.ReLU(inplace=True)) for c in [256,512,1024,1024]])
        self.combine=nn.Sequential(nn.Conv2d(4*d,d,3,padding=1,bias=False),nn.BatchNorm2d(d),nn.ReLU(inplace=True))
        self.bone_attention=ChannelSpatialAttention(d)
        self.final=nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.GELU())
        self.grid=grid
    def forward(self,x,attention=True):
        x=x.repeat(1,3,1,1)
        f=self.backbone
        x=f.pool0(f.relu0(f.norm0(f.conv0(x))))
        x=f.denseblock1(x);a=x;x=f.transition1(x)
        x=f.denseblock2(x);b=x;x=f.transition2(x)
        x=f.denseblock3(x);c=x;x=f.transition3(x)
        x=f.denseblock4(x);d=F.relu(f.norm5(x))
        mult=[F.adaptive_avg_pool2d(p(z),(self.grid,self.grid)) for z,p in zip([a,b,c,d],self.proj)]
        feat=self.combine(torch.cat(mult,dim=1))
        if attention:feat,spmap=self.bone_attention(feat)
        else:spmap=torch.ones_like(feat[:,:1])
        vector=self.final(F.adaptive_avg_pool2d(feat,1).flatten(1))
        return vector,feat,spmap,d

class WeightedPool(nn.Module):
    def __init__(self,d):
        super().__init__();self.score=nn.Sequential(nn.Linear(d,d//2),nn.Tanh(),nn.Linear(d//2,1))
    def forward(self,x):
        w=self.score(x).softmax(dim=1);return (w*x).sum(1),w.squeeze(-1)

class SpectralEncoder(nn.Module):
    # Ten subbands ordered LL3, LH3, HL3, HH3, LH2, HL2, HH2, LH1, HL1, HH1.
    def __init__(self,d=256,layers=4,heads=8,dropout=.1,pool=8):
        super().__init__();self.project=nn.ModuleList([nn.Linear(pool*pool,d) for _ in range(10)])
        self.pool=pool;self.scale=nn.Embedding(4,d);self.orientation=nn.Embedding(4,d)
        self.position=nn.Parameter(torch.zeros(1,11,d));nn.init.trunc_normal_(self.position,std=.02)
        self.cls=nn.Parameter(torch.zeros(1,1,d));nn.init.trunc_normal_(self.cls,std=.02)
        self.firstpool=WeightedPool(d)
        layer=nn.TransformerEncoderLayer(d,heads,dim_feedforward=4*d,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.finalpool=WeightedPool(d)
        self.out=nn.Sequential(nn.Linear(3*d,d),nn.LayerNorm(d),nn.GELU())
        self.register_buffer('scale_ids',torch.tensor([3,3,3,3,2,2,2,1,1,1]),persistent=False)
        self.register_buffer('orientation_ids',torch.tensor([0,1,2,3,1,2,3,1,2,3]),persistent=False)
    def forward(self,bands,use_transformer=True):
        # B,10,16,16. Adaptive pool avoids flattening mismatched wavelet levels.
        B=bands.size(0)
        z=F.adaptive_avg_pool2d(bands.reshape(B*10,1,*bands.shape[-2:]),(self.pool,self.pool)).flatten(1).reshape(B,10,-1)
        tok=torch.stack([p(z[:,i]) for i,p in enumerate(self.project)],1)
        tok=tok+self.scale(self.scale_ids)[None]+self.orientation(self.orientation_ids)[None]
        first,_=self.firstpool(tok)
        sequence=torch.cat([self.cls.expand(B,-1,-1),tok],1)+self.position
        ctx=self.transformer(sequence) if use_transformer else sequence
        weighted,importance=self.finalpool(ctx[:,1:])
        vec=self.out(torch.cat([ctx[:,0],weighted,first],1))
        return vec,ctx[:,1:],importance

class CrossFusion(nn.Module):
    def __init__(self,d=256,n_queries=8,n_heads=8,dropout=.1):
        super().__init__();self.n_queries=n_queries;self.d=d
        self.queries=nn.Linear(d,n_queries*d);self.normq=nn.LayerNorm(d);self.normkv=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,n_heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*d,d))
        self.norm2=nn.LayerNorm(d);self.pool=WeightedPool(d)
        self.out=nn.Sequential(nn.Linear(3*d,d),nn.LayerNorm(d),nn.GELU())
    def forward(self,spatial,spectral,memory,use_cross=True):
        B=spatial.size(0);query=self.queries(spatial).view(B,self.n_queries,self.d)
        if use_cross:
            retrieved,weights=self.attn(self.normq(query),self.normkv(memory),self.normkv(memory),need_weights=True,average_attn_weights=False)
            z=self.norm1(query+retrieved)
        else:
            weights=torch.zeros(B,self.attn.num_heads,self.n_queries,memory.size(1),device=spatial.device)
            z=self.norm1(query)
        z=self.norm2(z+self.ff(z));pool,_=self.pool(z)
        out=self.out(torch.cat([pool,spatial,spectral],dim=1))
        return out,weights

class OsteoXAINet(nn.Module):
    def __init__(self,pretrained=True,d=256,layers=4,heads=8,n_queries=8,cross_heads=8,transformer_dropout=.1,head_dropout=(.30,.20),num_classes=1,ablation='full'):
        super().__init__();self.ablation=ablation
        self.spatial=SpatialEncoder(d,pretrained=pretrained)
        self.spectral=SpectralEncoder(d,layers,heads,transformer_dropout)
        self.cross=CrossFusion(d,n_queries,cross_heads,transformer_dropout)
        self.spatial_head=nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.GELU())
        self.head=nn.Sequential(nn.Linear(d,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(head_dropout[0]),nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(head_dropout[1]),nn.Linear(128,num_classes))
    def forward(self,image,wavelets=None,return_maps=False):
        sp,feat,sa,last=self.spatial(image,attention=self.ablation not in ('backbone','bone','spectral','transformer'))
        if self.ablation in ('backbone','bone'):
            fused=self.spatial_head(sp); weights=None; importance=None
        else:
            if wavelets is None:raise ValueError('wavelets tensor is required for spectral models')
            spec,mem,importance=self.spectral(wavelets,use_transformer=self.ablation not in ('spectral',))
            if self.ablation in ('spectral','transformer','attention'):
                # For progressive ablations: spectral information included without cross-attention.
                fused=self.cross.out(torch.cat([sp,sp,spec],dim=1));weights=None
            else:fused,weights=self.cross(sp,spec,mem)
        logits=self.head(fused)
        if return_maps:return {'logits':logits,'spatial_map':sa,'feature_map':last,'cross_attention':weights,'spectral_importance':importance,'fused':fused}
        return logits

def set_mc_dropout(model,enabled=True):
    model.eval()
    if enabled:
        for m in model.head.modules():
            if isinstance(m,nn.Dropout):m.train()


def attention_entropy_penalty(weights,margin=1.4):
    if weights is None:return None
    probs=weights.clamp_min(1e-9)
    entropy=-(probs*probs.log()).sum(-1).mean()
    return F.relu(torch.as_tensor(margin,device=weights.device)-entropy)
