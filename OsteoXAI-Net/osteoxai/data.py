from __future__ import annotations
import os, json, hashlib, random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

LABELS={'normal':0,'healthy':0,'osteoporosis':1,'osteoporotic':1,'osteopenia':2,'osteopenic':2}
EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}

def inventory(root, output=None):
    root=Path(root)
    if not root.exists(): raise FileNotFoundError(f'Dataset root missing: {root}. Download the public dataset and extract it first.')
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in EXTS: continue
        labels=[LABELS[s.lower().replace(' ','').replace('_','').replace('-','')] for s in p.parts if s.lower().replace(' ','').replace('_','').replace('-','') in LABELS]
        if not labels: continue
        try:
            with Image.open(p) as im: im.verify()
            valid=True
        except Exception: valid=False
        rows.append({'path':str(p.resolve()),'label':labels[-1],'class_name':['Normal','Osteoporosis','Osteopenia'][labels[-1]],'valid':valid,'image_id':hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:16]})
    df=pd.DataFrame(rows,columns=['path','label','class_name','valid','image_id'])
    if output: Path(output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(output,index=False)
    return df

def fixed_splits(df,seed=42):
    df=df.loc[(df.label.isin([0,1])) & df.valid.astype(bool)].reset_index(drop=True)
    if len(df)<8 or df.label.value_counts().min()<3: raise ValueError('Need >=3 samples in each of the two classes for stratified splitting.')
    tr,temp=train_test_split(np.arange(len(df)),test_size=.30,stratify=df.label,random_state=seed)
    val,te=train_test_split(temp,test_size=.5,stratify=df.label.iloc[temp],random_state=seed)
    out=df.copy();out['split']='train';out.loc[val,'split']='val';out.loc[te,'split']='test'
    return out

def repeated_folds(df,repeats=5,folds=5,seeds=(42,123,2024,3407,7777)):
    df=df[df.label.isin([0,1]) & df.valid.astype(bool)].reset_index(drop=True)
    for rep in range(repeats):
        sk=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seeds[rep%len(seeds)])
        for fold,(tr,te) in enumerate(sk.split(np.arange(len(df)),df.label)):
            yield rep,fold,tr,te

def split_inner(df,index,seed=42,n_val=10):
    labs=df.iloc[index].label.to_numpy()
    n_val=min(n_val,len(index)-2)
    tr,va=train_test_split(index,test_size=n_val,random_state=seed,stratify=labs)
    return np.asarray(tr),np.asarray(va)

def read_gray(path,size=224):
    x=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if x is None: raise ValueError(f'Cannot read image: {path}')
    return cv2.resize(x,(size,size),interpolation=cv2.INTER_CUBIC)

def bone_contrast(image):
    # A rough bone-dominant ROI; not a validated anatomical segmentation.
    blur=cv2.GaussianBlur(image,(5,5),0)
    _,mask=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    pixels=image[mask>0]
    if len(pixels)<image.size*.05: pixels=image.ravel()
    pix=pixels.astype(np.float32)/255.0
    return float(pix.std()/(pix.mean()+1e-6))

def enhance(image,stats):
    con=bone_contrast(image)
    lo,hi=stats.get('contrast_quantiles',[0.,1.]); amin,amax=stats.get('clahe_range',[1.,4.])
    ratio=np.clip((con-lo)/(hi-lo+1e-8),0,1)
    clip=float(amax-(amax-amin)*ratio)
    clahe=cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8))
    return cv2.medianBlur(clahe.apply(image),3)

def augment(image,rng):
    h,w=image.shape
    angle=rng.uniform(-10,10); scale=rng.uniform(.95,1.05)
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,scale)
    M[0,2]+=rng.uniform(-.05,.05)*w;M[1,2]+=rng.uniform(-.05,.05)*h
    image=cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    if rng.random()<.5: image=cv2.flip(image,1)
    image=np.clip(image.astype(np.float32)*rng.uniform(.95,1.05)+rng.uniform(-7,7),0,255).astype(np.uint8)
    return image

def fit_preprocessing(paths,size=224):
    con=[];images=[]
    for p in paths:
        img=read_gray(p,size);images.append(img);con.append(bone_contrast(img))
    if not images: raise ValueError('Cannot fit preprocessing with zero training images.')
    st={'size':size,'contrast_quantiles':np.quantile(con,[.05,.95]).tolist(),'clahe_range':[1.,4.]}
    arr=np.stack([enhance(x,st) for x in images]).astype(np.float32)/255.
    st['mean']=float(arr.mean());st['std']=float(max(arr.std(),1e-5))
    # Wavelet train statistics: computed after image normalization, before train augmentations.
    sums=None;sq=None;cnt=None
    for x in arr:
        waves=wavelet_bands((x-st['mean'])/st['std'])
        if sums is None:
            sums=np.zeros(10,dtype=np.float64);sq=sums.copy();cnt=sums.copy()
        for i,b in enumerate(waves):
            sums[i]+=b.sum();sq[i]+=(b.astype(np.float64)**2).sum();cnt[i]+=b.size
    means=sums/cnt;stds=np.sqrt(np.maximum(sq/cnt-means**2,1e-8))
    st['wavelet_mean']=means.tolist();st['wavelet_std']=stds.tolist()
    return st

# DB4 exact filter coefficients for environments where PyWavelets is not installed.
_DB4_LO=np.array([-0.010597401785069032,0.0328830116668852,0.030841381835560764,-0.18703481171888114,-0.027983769416859854,0.6308807679298587,0.7148465705529154,0.2303778133088964])
_DB4_HI=np.array([(-1)**i * _DB4_LO[::-1][i] for i in range(8)])

def _dwt2_db4_torch(arr):
    import torch.nn.functional as F
    lo=torch.from_numpy(_DB4_LO.copy()).float();hi=torch.from_numpy(_DB4_HI.copy()).float()
    # convolution in torch is correlation; reverse analysis filters.
    kernels=torch.stack([torch.outer(a,b) for a,b in [(lo,lo),(lo,hi),(hi,lo),(hi,hi)]])[:,None].flip((-1,-2))
    x=torch.tensor(np.asarray(arr).copy(),dtype=torch.float32)[None,None]
    x=F.pad(x,(3,4,3,4),mode='reflect')
    y=F.conv2d(x,kernels,stride=2)[0].numpy()
    return y[0],(y[1],y[2],y[3])

def wavelet_bands(img):
    try:
        import pywt
        co=pywt.wavedec2(img,'db4',level=3,mode='symmetric')
        return [np.asarray(co[0],dtype=np.float32)]+[np.asarray(z,dtype=np.float32) for group in co[1:] for z in group]
    except ImportError:
        a=img;groups=[]
        for _ in range(3):
            a,detail=_dwt2_db4_torch(a);groups.insert(0,detail)
        return [a]+[z for g in groups for z in g]

class RadiographDataset(Dataset):
    def __init__(self,df,stats,training=False,seed=42):
        self.df=df.reset_index(drop=True);self.stats=stats;self.training=training;self.seed=seed;self.epoch=0
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i];im=read_gray(r.path,self.stats['size']);im=enhance(im,self.stats)
        if self.training:
            worker=torch.utils.data.get_worker_info()
            rng=random.Random(self.seed+100003*self.epoch+i+1009*(worker.id if worker else 0))
            im=augment(im,rng)
        img=(im.astype(np.float32)/255.-self.stats['mean'])/self.stats['std']
        bands=wavelet_bands(img)
        bands=[(x-m)/max(s,1e-5) for x,m,s in zip(bands,self.stats['wavelet_mean'],self.stats['wavelet_std'])]
        # Align spatial dimensions for efficient batched handling while preserving individual scales via adaptive pooling.
        bands=np.stack([cv2.resize(x,(16,16),interpolation=cv2.INTER_AREA) for x in bands]).astype(np.float32)
        return {'image':torch.from_numpy(img[None]).float(),'wavelets':torch.from_numpy(bands),'label':torch.tensor(float(r.label)), 'image_id':str(r.image_id),'path':str(r.path)}

def phash_aug(path):
    im=read_gray(path,64); hashes=[]
    for z in (im,cv2.flip(im,1)):
        for k in range(4):
            t=np.rot90(z,k).copy();dct=cv2.dct(cv2.resize(t,(32,32)).astype('float32'))[:8,:8]
            bits=(dct>np.median(dct[1:])).ravel();hashes.append(int(''.join(map(lambda v:'1' if v else '0',bits)),2))
    return hashes

def deduplicate(reference,candidates,out_csv,phash_max=8,embedding_min=.95,device='cpu',pretrained=True):
    """Flag overlaps using augmented pHash OR DenseNet121 embedding cosine, with review audit."""
    from torchvision.models import densenet121,DenseNet121_Weights
    from torch.nn import functional as F
    weights=DenseNet121_Weights.DEFAULT if pretrained else None
    net=densenet121(weights=weights).features.to(device).eval()
    paths=list(reference.path)+list(candidates.path)
    hs=[phash_aug(p) for p in paths]
    def embed(p):
        g=read_gray(p,224);g=cv2.resize(g,(224,224)).astype(np.float32)/255
        t=torch.from_numpy(np.stack([g]*3))[None].to(device)
        with torch.no_grad():
            f=F.adaptive_avg_pool2d(F.relu(net(t)),1).flatten(1)
            return F.normalize(f,dim=1).cpu().numpy()[0]
    em=np.stack([embed(p) for p in paths]);nr=len(reference)
    # Also remove duplicated candidates; deterministic first-image retention.
    records=[];kept=[]
    for j in range(nr,len(paths)):
        match=None;reason='';hd=64;sim=-1.
        for i in list(range(nr))+kept:
            d=min((a^b).bit_count() for a in hs[i] for b in hs[j]);s=float(em[i]@em[j])
            if d<=phash_max or s>=embedding_min:
                match=i;reason=('phash' if d<=phash_max else 'embedding');hd=d;sim=s;break
        if match is None: kept.append(j)
        records.append({'path':paths[j],'suspected_duplicate':match is not None,'reference_path':paths[match] if match is not None else '', 'reason':reason,'hamming':hd if match is not None else '', 'cosine':sim if match is not None else '', 'review_status':'requires_visual_review' if match is not None else 'not_flagged'})
    df=pd.DataFrame(records);Path(out_csv).parent.mkdir(parents=True,exist_ok=True);df.to_csv(out_csv,index=False)
    # Conservative default: do not claim automatic matching is validated; flagged images excluded pending review.
    retain=set(df.loc[~df.suspected_duplicate,'path']);return candidates[candidates.path.isin(retain)].copy(),df
