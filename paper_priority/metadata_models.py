"""Semantic SVC with inductively fitted individual-author and venue features."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.svm import SVC
from .advanced import author_tokens
from .semantic_models import SemanticHead

META_HEADS = {f'{encoder}_authors{int(a*100)}':
              dict(encoder=encoder, kind='svc', C=4., authors=a, venue=.35, numeric=.05, metadata=True)
              for encoder in ('bge','e5base') for a in (0., .35, .75)}


def nonempty_authors(text):
    return [] if text == '_missing_' else author_tokens(text)


class MetadataSVC(SemanticHead):
    def fit(self, embeddings, x, y):
        self.center_ = np.asarray(embeddings,dtype=float).mean(axis=0)
        self.venue_ = OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(x[:,[1]])
        self.authors_ = None
        if self.spec['authors']:
            vectorizer = TfidfVectorizer(tokenizer=nonempty_authors,token_pattern=None,lowercase=False,min_df=2,max_features=6000)
            try:
                vectorizer.fit(x[:,2])
                self.authors_ = vectorizer
            except ValueError as exc:
                if 'empty vocabulary' not in str(exc) and 'no terms remain' not in str(exc).lower():
                    raise
        self.model_ = SVC(C=self.spec['C'],gamma='scale',probability=True,random_state=self.seed).fit(self.transform(embeddings,x),y)
        return self

    def transform(self, embeddings, x):
        blocks = [normalize(np.asarray(embeddings,dtype=float)-self.center_),
                  self.venue_.transform(x[:,[1]])*self.spec['venue'],
                  np.asarray(x[:,[3,4,5,6]],dtype=float)*self.spec['numeric']]
        if self.authors_ is not None:
            blocks.append(self.authors_.transform(x[:,2]).toarray()*self.spec['authors'])
        return np.column_stack(blocks)
