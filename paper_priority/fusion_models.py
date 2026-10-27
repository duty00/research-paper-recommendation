"""Joint sparse+dense ordinal models, preserving earlier versions."""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from .advanced import make_features, fit_binary, positive_probability, AdvancedModel
from .semantic_models import SemanticHead

SPECS={}
for encoder in ('e5','e5base','bge'):
    for name,profile,scale,C in [('joint_s1_c1','title',1.,1.),('joint_s2_c1','title',2.,1.),
                                ('joint_s2_c4','title',2.,4.),('joint_rich_s2_c1','rich',2.,1.)]:
        SPECS[f'{encoder}_{name}']=dict(mode='joint',encoder=encoder,profile=profile,scale=scale,C=C)
for encoder in ('e5base','bge'):
    for name,kind,metadata in [('ordinal_c4','ordinal',False),('ordinal_meta_c4','ordinal',True),('svc_c4','svc',False)]:
        SPECS[f'{encoder}_{name}']=dict(mode='semantic',encoder=encoder,kind=kind,metadata=metadata,C=4.)

WEIGHTS={name:{name:1.} for name in SPECS}
WEIGHTS.update({f'blend_{name}_v3':{name:.5,'old_v3':.5} for name in SPECS})


class JointOrdinal(BaseEstimator):
    def __init__(self,spec,seed=261):
        self.spec=spec
        self.seed=seed

    def matrix(self,embeddings,words):
        dense=normalize(np.asarray(embeddings,dtype=float)-self.center_)*self.spec['scale']
        return sparse.hstack([words,sparse.csr_matrix(dense)],format='csr')

    def fit(self,embeddings,x,y):
        features=make_features(self.spec['profile'])
        words=features.fit_transform(x)
        return self.fit_transformed(features,words,embeddings,y)

    def fit_transformed(self,features,words,embeddings,y):
        self.features_=features
        self.center_=np.asarray(embeddings,dtype=float).mean(axis=0)
        matrix=self.matrix(embeddings,words)
        y=np.asarray(y,dtype=int)
        self.models_=[fit_binary(matrix,(y>boundary).astype(int),self.spec['C'],self.seed) for boundary in (1,2,3,4)]
        return self

    def predict_transformed(self,embeddings,words):
        matrix=self.matrix(embeddings,words)
        thresholds=np.column_stack([positive_probability(m,matrix) for m in self.models_])
        return np.column_stack([thresholds.mean(axis=1),thresholds[:,0]])

    def predict_components(self,embeddings,x):
        return self.predict_transformed(embeddings,self.features_.transform(x))


class V3Baseline(BaseEstimator):
    def fit(self,embeddings,x,y):
        self.words_=AdvancedModel(dict(profile='title',kind='ordinal',C=1.),261).fit(x,y)
        self.semantic_=SemanticHead(dict(encoder='e5',kind='ordinal',C=4.,metadata=True),261).fit(embeddings,x,y)
        return self

    def predict_components(self,embeddings,x):
        return .5*self.words_.predict_components(x)+.5*self.semantic_.predict_components(embeddings,x)
