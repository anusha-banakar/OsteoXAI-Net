import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,roc_auc_score,average_precision_score,confusion_matrix,brier_score_loss,roc_curve)
from scipy.stats import beta, binomtest

def ece(y,p,bins=10):
    y=np.asarray(y);p=np.asarray(p);pred=(p>=.5).astype(int);conf=np.maximum(p,1-p);total=0.
    for a,b in zip(np.linspace(.5,1,bins+1)[:-1],np.linspace(.5,1,bins+1)[1:]):
        mask=(conf>=a)&(conf<=b if b==1 else conf<b)
        if mask.any(): total+=mask.mean()*abs((pred[mask]==y[mask]).mean()-conf[mask].mean())
    return float(total)

def classification_metrics(y,p,threshold=.5):
    y=np.asarray(y).astype(int);p=np.asarray(p).astype(float);yp=(p>=threshold).astype(int)
    cm=confusion_matrix(y,yp,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    out={'n':len(y),'accuracy':accuracy_score(y,yp),'precision':precision_score(y,yp,zero_division=0),'recall':recall_score(y,yp,zero_division=0),'specificity':tn/max(1,tn+fp),'f1':f1_score(y,yp,zero_division=0),'mcc':matthews_corrcoef(y,yp),'ece':ece(y,p),'brier':brier_score_loss(y,p),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'threshold':float(threshold)}
    out['roc_auc']=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None
    out['pr_auc']=float(average_precision_score(y,p)) if len(np.unique(y))>1 else None
    return out

def select_threshold(y,p):
    # Validation-only Youden J. Use ROC-AUC invariant to threshold, optimizing balance.
    if len(np.unique(y))<2:return .5
    fpr,tpr,thresholds=roc_curve(y,p)
    valid=np.isfinite(thresholds)
    j=np.where(valid,tpr-fpr,-np.inf)
    return float(np.clip(thresholds[np.argmax(j)],.05,.95))

def bootstrap_ci(y,p,func,n_boot=2000,seed=42,alpha=.05):
    y=np.asarray(y);p=np.asarray(p);rng=np.random.default_rng(seed);groups=[np.where(y==cl)[0] for cl in np.unique(y)]
    results=[]
    for _ in range(n_boot):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        try:results.append(float(func(y[idx],p[idx])))
        except ValueError:continue
    return [float(x) for x in np.quantile(results,[alpha/2,1-alpha/2])] if results else [None,None]

def exact_ci(k,n,alpha=.05):
    return (float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.)

def risk_coverage(y,p,score,threshold=.5):
    y=np.asarray(y);p=np.asarray(p);score=np.asarray(score)
    order=np.argsort(-score);err=((p>=threshold).astype(int)!=y)[order]
    coverage=np.arange(1,len(y)+1)/len(y);risk=np.cumsum(err)/np.arange(1,len(y)+1)
    return {'coverage':coverage.tolist(),'risk':risk.tolist(),'aurc':float(np.trapezoid(risk,coverage))}

def decision_curve(y,p,thresholds=np.arange(.05,.51,.05)):
    y=np.asarray(y);p=np.asarray(p);n=len(y)
    return [{'threshold':float(t),'model':float((((p>=t)&(y==1)).sum()-((p>=t)&(y==0)).sum()*t/(1-t))/n),'all':float((y.sum()-(n-y.sum())*t/(1-t))/n),'none':0.} for t in thresholds]

def mcnemar_exact(a,b,y,threshold=.5):
    ca=(np.asarray(a)>=threshold)==np.asarray(y);cb=(np.asarray(b)>=threshold)==np.asarray(y)
    n01=int((ca&~cb).sum());n10=int((~ca&cb).sum());n=n01+n10
    return {'a_only_correct':n01,'b_only_correct':n10,'p':float(binomtest(min(n01,n10),n,.5).pvalue) if n else 1.}
