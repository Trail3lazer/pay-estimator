from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from typing import Callable
import logging, os, settings, json

class Categorizer:
    def __init__(self, model_vectors: KeyedVectors, tokenize: Callable[[str],list[str]], verbosity=0):
        self.wv = model_vectors
        self.tokenize = tokenize
        self.kv = KeyedVectors(self.wv.vector_size)
        self.default_category:tuple = (None, np.nan)
        self.verbosity=verbosity
        #self.category_vectors: pd.Series[list[float]] = None
        self.category_labels: list[str] = None
        self.categories_created = False
        

  
    def get_similar_categories(self, sentence: str, topn=5) -> list[tuple[str,float]]:
        tkns = []
        if isinstance(sentence, str):
            tkns = self.tokenize(sentence)
        if len(tkns) == 0:
            return [self.default_category]
        vec = self.wv.get_mean_vector(tkns)
        categories = self.get_vecs().similar_by_vector(vec, topn)
        return categories



    def categorize(self, sentence: str) -> tuple:
        similar = self.get_similar_categories(sentence, topn=1)
        if len(similar) == 0:
            print('This did not match: ' + sentence)
            return self.default_category
        return similar[0]
    
    
    
    def categorize_to_vec(self, sentence: str) -> tuple:
        try:
            similar = self.get_similar_categories(sentence, topn=1)
            if len(similar) == 0:
                raise Exception('No match')
        except Exception as e:
            return 
        return self.get_vecs().get_vector(similar[0][0])
    
    
    
    def categorize_list(self, sentences: list[str]) -> dict[str, list[tuple[str, float]]]:
        result = []
        for sentence in np.array(sentences):
            category = self.default_category
            if isinstance(sentence, str) and len(sentence) > 0:
                category = self.categorize(sentence)
                result.append((sentence, category[0]))
            else:
                result.append((sentence, None))
            
        return result
    
    
    
    def get_vecs(self):
        if self.categories_created:
            return self.kv
        
        print('Creating categories.')
        categories = dict(json.load(open(settings.JOB_FIELDS)))

        print('Creating KeyedVectors from the category names.')
        keys = []
        vectors = []
        for c in categories:
            if not isinstance(c, str) or len(c) == 0:
                continue
            tkns = self.tokenize(categories[c])
            if not isinstance(tkns, list) or len(tkns) == 0:
                if self.verbosity: logging.warning(f'Category is empty after tokenization so it will not be included. "{c}"')
                continue
            vec = self.wv.get_mean_vector(tkns)
            vectors.append(vec)
            keys.append(c)
        if len(keys) == 0:
            raise ValueError('Could not create vectors for any input categories.')
        
        self.kv.add_vectors(keys, vectors)
        
        self.categories_created = True
        
        return self.kv

        
        
        