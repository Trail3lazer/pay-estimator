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
        self.default_category:str = None
        self.verbosity=verbosity
        self.category_vectors: pd.Series[list[float]] = None
        self.category_labels: pd.Series[str] = None
        self.categories_created = False
        
        
        
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
        self._category_vectors = pd.Series(vectors, index=keys)
        

    
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
    
    
    
    def create_category_vectors(self):

        if os.path.isfile(settings.CATEGORY_VECS):
            print("Retrieving category vectors from "+settings.CATEGORY_VECS)
            self.replace_vectors(KeyedVectors.load(settings.CATEGORY_VECS))
            return
        
        print('Creating categories.')
        categories = self.get_category_labels()

        print('Creating KeyedVectors from the category names.')
        self.add_categories(categories)

        print('Saving the KeyedVectors.')
        self.kv.save(settings.CATEGORY_VECS)
        
        self.categories_created = True
    
    
    
    def get_category_labels(self) -> list[str]:
        
        if self.category_labels:
            return self.category_labels
        
        if os.path.isfile(settings.JOB_CATEGORIES):
            print("Retrieving category labels from "+settings.JOB_CATEGORIES)
            self.category_labels = pd.read_json(settings.JOB_CATEGORIES, typ='series', orient='values')
            return self.category_labels
        
        print('Creating categories.')
        jobs = json.load(open(settings.BLS_JOBS))
        
        groups: dict[str, list[str]] = {}
        for x in jobs:
            category = x[0]
            title = None
            if len(x) > 1:
                category = x[1]
                title = x[0]
            if not category in groups:
                 groups[category] = []
            if title:
                groups[category].append(title)

        self.category_labels = pd.Series(groups.keys())
        
        self.category_labels.to_json(settings.JOB_CATEGORIES, indent=2)
        
        return self.category_labels
        
    
    
    def get_category_vectors(self):
        if os.path.isfile(settings.CATEGORY_VECS):
            print("Retrieving category labels from "+settings.CATEGORY_VECS)
            self.category_labels = pd.read_json(settings.CATEGORY_VECS)
            
            return self.category_labels
        
        labels = self.get_category_labels()
        if not self.categories_created:
            self.create_category_vectors()
        vecs = [self.kv.get_vector(name) for name in labels]
        cat_vecs = pd.DataFrame(vecs, index=labels)
        
        
        