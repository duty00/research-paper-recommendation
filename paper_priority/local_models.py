"""Local semantic regressors for gain and probability of a worthwhile paper."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.metrics.pairwise import euclidean_distances

HEADS={}
for encoder in ('bge','e5base'):
    for gamma in (1.,2.):
        for alpha in (.1,1.,10.):
            HEADS[f'{encoder}_krr_g{gamma:g}_a{alpha:g}']=dict(
                encoder=encoder,kind='kernel',gamma=gamma,alpha=alpha)
    for neighbors in (30,100):
        HEADS[f'{encoder}_neighbors{neighbors}']=dict(
            encoder=encoder,kind='neighbors',neighbors=neighbors,temperature=.15,smoothing=5.)


class LocalSemanticHead(BaseEstimator):
    def __init__(self,spec):
        self.spec=spec

    def transform(self,embeddings,x):
        return np.column_stack([
            normalize(np.asarray(embeddings,dtype=float)-self.center_),
            .35*self.venue_.transform(x[:,[1]]),
            .05*np.asarray(x[:,[3,4,5,6]],dtype=float)])

    def fit(self,embeddings,x,y):
        y=np.asarray(y,dtype=int)
        if len(y)!=len(x) or not np.isin(y,[1,2,3,4,5]).all():
            raise ValueError('Expected one label in 1..5 per training row.')
        self.center_=np.asarray(embeddings,dtype=float).mean(axis=0)
        self.venue_=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(x[:,[1]])
        features=self.transform(embeddings,x)
        targets=np.column_stack([(y-1)/4,(y>1).astype(float)])
        self.prior_=targets.mean(axis=0)
        if self.spec['kind']=='kernel':
            self.model_=KernelRidge(alpha=self.spec['alpha'],kernel='rbf',gamma=self.spec['gamma'])
            self.model_.fit(features,targets-self.prior_)
        elif self.spec['kind']=='neighbors':
            self.features_=features
            self.targets_=targets
        else:
            raise ValueError('Unknown local head kind.')
        return self

    def predict_components(self,embeddings,x):
        features=self.transform(embeddings,x)
        if self.spec['kind']=='kernel':
            result=self.model_.predict(features)+self.prior_
        else:
            distances=euclidean_distances(features,self.features_,squared=True)
            k=min(self.spec['neighbors'],len(self.features_))
            # Stable sorting keeps equal-distance neighbors deterministic.
            indices=np.argsort(distances,axis=1,kind='stable')[:,:k]
            local=np.take_along_axis(distances,indices,axis=1)
            weights=np.exp(-(local-local[:,[0]])/(2*self.spec['temperature']))
            smooth=self.spec['smoothing']
            result=(np.sum(self.targets_[indices]*weights[:,:,None],axis=1)+smooth*self.prior_)/(weights.sum(axis=1)[:,None]+smooth)
        result=np.clip(result,0.,1.)
        result[:,0]=np.minimum(result[:,0],result[:,1])
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite local semantic predictions.')
        return result
