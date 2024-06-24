from gensim.models import KeyedVectors
import numpy as np
from typing import Callable

class Categorizer:
    def __init__(self, model_vectors: KeyedVectors, categories: dict[str, str], tokenize: Callable[[str],list[str]]):
        self.wv = model_vectors
        self.tokenize = tokenize
        self.default_category = ''
        keys = []
        vectors = []
        for k,v in categories:
            tkns = tokenize(k+' '+v)
            vec = self.wv.get_mean_vector(tkns)
            vectors.append(vec)
            keys.append(k)
        self.kv = KeyedVectors(self.wv.vector_size)
        self.kv.add_vectors(keys, vectors)
        
        
        
    def get_similar_categories(self, sentence: str, topn=5) -> list[tuple[str,float]]:
        if not isinstance(sentence, str):
            raise ValueError('Cannot calculate similar categories for missing strings.')
        if len(sentence) == 0:
            return []
        tkns = self.tokenize(sentence)
        vec = self.wv.get_mean_vector(tkns)
        categories = self.kv.similar_by_vector(vec, topn)
        return categories



    def categorize(self, sentence: list[str])-> tuple:
        similar = self.get_similar_categories(sentence, topn=1)
        if len(similar) == 0:
            return (self.default_category, np.nan)
        return similar[0]
    
    
    
    def categorize_list(self, senteces: list[str]) -> dict[str, list[tuple[str, str, float]]]:
        groups: dict[str, list[tuple[str, str, float]]] = {}
        for sentence in senteces:
            category = (self.default_category, np.nan)
            if isinstance(sentence, str) and len(sentence) > 0:
                category = self.categorize(sentence)
            key = category[0]
            if not key in groups:
                groups[key] = []
            groups[key].append((sentence, category[0], category[1]))
        return groups