"""Small fixed SVC regularization/kernel experiment around v4."""
import numpy as np
from sklearn.svm import SVC
from .semantic_models import SemanticHead

NEW_SPECS={f'{encoder}_{name}':dict(encoder=encoder,C=C,gamma_factor=gamma,kind='svc',metadata=False)
           for encoder in ('bge','e5base')
           for name,C,gamma in [('c1_g1',1.,1.),('c16_g1',16.,1.),('c4_g05',4.,.5),('c4_g2',4.,2.)]}


class RefinedSVC(SemanticHead):
    def fit(self,embeddings,x,y):
        self.center_=np.asarray(embeddings,dtype=float).mean(axis=0)
        features=self.transform(embeddings,x)
        gamma=self.spec['gamma_factor']/(features.shape[1]*features.var())
        self.model_=SVC(C=self.spec['C'],gamma=gamma,probability=True,random_state=self.seed).fit(features,y)
        return self


def candidate_specs():
    candidates={}
    # Fixed .25 simplex grid over the three distinct prediction sources.
    for b in range(5):
        for e in range(5-b):
            v=4-b-e
            weights={k:q/4 for k,q in [('bge_svc_c4',b),('e5base_svc_c4',e),('old_v3',v)] if q}
            for aggregation in ('probability','rank'):
                name=f'{aggregation}_b{b}_e{e}_v{v}'
                candidates[name]=dict(weights=weights,aggregation=aggregation,family=f'ensemble_{aggregation}')
    for name in NEW_SPECS:
        for q in (.25,.5,.75):
            weights={name:q,'bge_svc_c4':(1-q)*.5,'old_v3':(1-q)*.5}
            candidates[f'{name}_mix{int(q*100)}']=dict(weights=weights,aggregation='probability',family=name)
    return candidates


def aggregate(predictions,spec):
    from scipy.stats import rankdata
    result=None
    for name,weight in spec['weights'].items():
        component=np.asarray(predictions[name],dtype=float).copy()
        if spec['aggregation']=='rank':
            component[:,0]=(rankdata(component[:,0],method='average')-1)/max(len(component)-1,1)
        elif spec['aggregation']!='probability':
            raise ValueError('Unknown aggregation.')
        result=weight*component if result is None else result+weight*component
    return result
