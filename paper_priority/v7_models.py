"""Local kernel/metadata refinements of the frozen V6 candidate2 ensemble."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from .metadata_models import MetadataSVC,nonempty_authors
from .metric import make_priorities

PARENT_SPEC=dict(weights={'bge_authors75':.25,'bge_svc_c4':.28125,'e5base_c4_g2':.1875,'old_v3':.28125},aggregation='probability')
HEADS={
    'bge_meta_c16_g1':dict(encoder='bge',C=16.,gamma_factor=1.,authors=.75,venue=.35,family='kernel'),
    'bge_meta_c4_g05':dict(encoder='bge',C=4.,gamma_factor=.5,authors=.75,venue=.35,family='kernel'),
    'bge_meta_c4_g2':dict(encoder='bge',C=4.,gamma_factor=2.,authors=.75,venue=.35,family='kernel'),
    'bge_meta_auth125':dict(encoder='bge',C=4.,gamma_factor=1.,authors=1.25,venue=.35,family='author'),
    'bge_meta_venue07':dict(encoder='bge',C=4.,gamma_factor=1.,authors=.75,venue=.7,family='venue'),
    'e5base_meta_c4_g2':dict(encoder='e5base',C=4.,gamma_factor=2.,authors=.75,venue=.35,family='encoder'),
}
HEADS={name:dict(spec,kind='svc',metadata=True,numeric=.05) for name,spec in HEADS.items()}


class TunedMetadataSVC(MetadataSVC):
    def fit(self,embeddings,x,y):
        self.center_=np.asarray(embeddings,dtype=float).mean(axis=0)
        self.venue_=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(x[:,[1]])
        self.authors_=None
        if self.spec['authors']:
            vectorizer=TfidfVectorizer(tokenizer=nonempty_authors,token_pattern=None,lowercase=False,min_df=2,max_features=6000)
            try:
                vectorizer.fit(x[:,2])
                self.authors_=vectorizer
            except ValueError as exc:
                if 'empty vocabulary' not in str(exc) and 'no terms remain' not in str(exc).lower():
                    raise
        features=self.transform(embeddings,x)
        variance=float(features.var())
        if variance<=0:
            raise ValueError('Degenerate features.')
        gamma=self.spec['gamma_factor']/(features.shape[1]*variance)
        self.model_=SVC(C=self.spec['C'],gamma=gamma,probability=True,random_state=self.seed).fit(features,y)
        return self


def candidates():
    result={'parent_policy':dict(**PARENT_SPEC,family='policy')}
    for name,head in HEADS.items():
        for q in (.25,.5,.75):
            weights={name:q,**{k:(1-q)*w for k,w in PARENT_SPEC['weights'].items()}}
            result[f'{name}_mix{int(q*100)}']=dict(weights=weights,aggregation='probability',family=head['family'])
    return result


def priorities(ids,components,fraction,alpha):
    ids=np.asarray(ids,dtype=str)
    components=np.asarray(components,dtype=float)
    if components.shape!=(len(ids),2) or not np.all(np.isfinite(components)) or np.any((components<0)|(components>1)):
        raise ValueError('Expected bounded finite Nx2 gain/read components.')
    if not 0<=alpha<=1:
        raise ValueError('Invalid reading-probability weight.')
    chosen=make_priorities(ids,components[:,0]+alpha*components[:,1],fraction)>0
    result=np.zeros(len(ids))
    result[chosen]=make_priorities(ids[chosen],components[chosen,0],1.)
    return result
