"""Author/venue preference models fitted only on each training fold."""
from collections import Counter,defaultdict
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from .advanced import fit_binary,positive_probability
from .metadata_models import nonempty_authors
from .v7_models import PARENT_SPEC

HEADS={
    **{f'author_ridge_a{int(a)}':dict(kind='ridge',alpha=a,family='ridge') for a in (1.,10.,100.)},
    'author_ordinal_c01':dict(kind='ordinal',C=.1,family='ordinal'),
    'author_ordinal_c1':dict(kind='ordinal',C=1.,family='ordinal'),
    'author_prior_s2':dict(kind='prior',smoothing=2.,family='prior'),
    'author_prior_s10':dict(kind='prior',smoothing=10.,family='prior'),
}


class AuthorPreference:
    def __init__(self,spec,seed=261):
        self.spec=spec
        self.seed=seed

    def fit(self,x,y):
        y=np.asarray(y,dtype=int)
        target=np.column_stack([(y-1)/4,(y>1).astype(float)])
        if self.spec['kind']=='prior':
            self.global_=target.mean(axis=0)
            venue_sum=defaultdict(lambda:np.zeros(2))
            venue_count=Counter()
            for row,value in zip(x,target):
                venue_sum[row[1]]+=value
                venue_count[row[1]]+=1
            self.venues_={v:(value+20*self.global_)/(venue_count[v]+20) for v,value in venue_sum.items()}
            author_sum=defaultdict(lambda:np.zeros(2))
            self.counts_=Counter()
            for row,value in zip(x,target):
                residual=value-self.venues_[row[1]]
                for author in set(nonempty_authors(row[2])):
                    author_sum[author]+=residual
                    self.counts_[author]+=1
            self.offsets_={a:value/(self.counts_[a]+self.spec['smoothing']) for a,value in author_sum.items()}
            return self
        self.venue_=OneHotEncoder(handle_unknown='ignore').fit(x[:,[1]])
        self.authors_=None
        vectorizer=TfidfVectorizer(tokenizer=nonempty_authors,token_pattern=None,lowercase=False,min_df=1,max_features=12000)
        try:
            vectorizer.fit(x[:,2])
            self.authors_=vectorizer
        except ValueError as exc:
            if 'empty vocabulary' not in str(exc): raise
        features=self.transform(x)
        if self.spec['kind']=='ridge':
            self.model_=Ridge(alpha=self.spec['alpha'],solver='lsqr',tol=1e-7).fit(features,target)
        else:
            self.models_=[fit_binary(features,(y>boundary).astype(int),self.spec['C'],self.seed) for boundary in (1,2,3,4)]
        return self

    def transform(self,x):
        blocks=[self.venue_.transform(x[:,[1]])*.5,sparse.csr_matrix(np.asarray(x[:,[3,4,5,6]],dtype=float)*.05)]
        if self.authors_ is not None:
            blocks.append(self.authors_.transform(x[:,2]))
        return sparse.hstack(blocks,format='csr')

    def predict_components(self,x):
        if self.spec['kind']=='prior':
            values=[]
            for row in x:
                authors=set(nonempty_authors(row[2]))
                offset=sum((self.offsets_.get(a,np.zeros(2)) for a in authors),np.zeros(2))/max(len(authors),1)
                values.append(self.venues_.get(row[1],self.global_)+offset)
            return np.clip(np.asarray(values),0,1)
        features=self.transform(x)
        if self.spec['kind']=='ridge':
            values=self.model_.predict(features)
        else:
            probabilities=np.column_stack([positive_probability(model,features) for model in self.models_])
            values=np.column_stack([probabilities.mean(axis=1),probabilities[:,0]])
        return np.clip(values,0,1)


def candidates():
    return {f'{name}_mix{int(round(q*100))}':dict(weights={name:q,**{k:(1-q)*w for k,w in PARENT_SPEC['weights'].items()}},
            aggregation='probability',family=spec['family'])
            for name,spec in HEADS.items() for q in (.1,.25,.5)}
