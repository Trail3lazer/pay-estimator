from gensim.models import KeyedVectors
import numpy as np
from typing import Callable
import logging

class Categorizer:
    def __init__(self, model_vectors: KeyedVectors, tokenize: Callable[[str],list[str]], verbosity=0):
        self.wv = model_vectors
        self.tokenize = tokenize
        self._kv = KeyedVectors(self.wv.vector_size)
        self.default_category = None
        self.verbosity=verbosity
        
    @property
    def kv(self):
        return self._kv    
        
        
        
    def add_categories(self, categories: list[list[str]]):
        keys = []
        vectors = []
        for category in categories:
            if not isinstance(category, list) or len(category) == 0:
                continue
            tkns = self.tokenize(' '.join(category))
            if not isinstance(tkns, list) or len(tkns) == 0:
                if self.verbosity: logging.warning(f'Category is empty after tokenization so it will not be included. "{category}"')
                continue
            vec = self.wv.get_mean_vector(tkns)
            vectors.append(vec)
            keys.append(' '.join(self.tokenize(category[0])))
        self.kv.add_vectors(keys, vectors)
        

    
    def replace_vectors(self, category_vectors: KeyedVectors):
        self._kv = category_vectors
        
  
        
    def get_similar_categories(self, sentence: str, topn=5) -> list[tuple[str,float]]:
        tkns = []
        if isinstance(sentence, str):
            tkns = self.tokenize(sentence)
        else:
            raise ValueError(f'Cannot calculate similar categories for missing strings. "{sentence}"')
        if len(tkns) == 0:
            raise ValueError(f'Sentence is empty after tokenization so it will not be included. "{sentence}"')
        vec = self.wv.get_mean_vector(tkns)
        categories = self.kv.similar_by_vector(vec, topn)
        return categories



    def categorize(self, sentence: str) -> tuple:
        similar = self.get_similar_categories(sentence, topn=1)
        if len(similar) == 0:
            print('This did not match: ' + sentence)
            return (self.default_category, np.nan)
        return similar[0]
    
    
    
    def categorize_list(self, sentences: list[str]) -> dict[str, list[tuple[str, float]]]:
        result = []
        for sentence in np.array(sentences):
            category = (self.default_category, np.nan)
            if isinstance(sentence, str) and len(sentence) > 0:
                category = self.categorize(sentence)
            else:
                print('This is not valid: ' + sentence)
            result.append((sentence, category[0]))
        return result
    
    
    
    def categorize_list_top(self, sentences: list[str]) -> dict[str, list[tuple[str, float]]]:
        result = []
        for sentence in np.array(sentences):
            category = (self.default_category, np.nan)
            if isinstance(sentence, str) and len(sentence) > 0:
                category = self.get_similar_categories(sentence, 3)
            else:
                print('This is not valid: ' + sentence)
            result.append((sentence, category[0], category[1], category[2]))
        return result