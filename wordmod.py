import os, settings, pickle
import pandas as pd
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

class callback(CallbackAny2Vec): # https://stackoverflow.com/questions/54888490/gensim-word2vec-print-log-loss
    def __init__(self):
        self.epoch = 0
        self.last_total = 0
        self.last_loss = 0

    def on_epoch_end(self, model):
        total = model.get_latest_training_loss()
        loss = total-self.last_total
        print(f'epoch: {self.epoch}, loss: {loss}, diff: {loss-self.last_loss}')
        self.epoch += 1
        self.last_total = total
        self.last_loss = loss

class Job2Vec:
    def __init__(self):
        self._jobs_df: pd.DataFrame = None
        self._dataset: pd.Series = None
        self._model = None
        self._default_vec: list[float] = None
    
    
    def try_load_dataset(self):
        if os.path.isfile(settings.TOKENIZED_JOBS):
            print("Retrieving an existing dataset at "+settings.TOKENIZED_JOBS)
            with open(settings.TOKENIZED_JOBS, 'rb') as f:
                dataset = pickle.load(f)
            return list(dataset)
        return None
    
    
    def preprocess_data(self, sentences: list[str]) -> np.ndarray:
        
        print("Cleaning and tokenizing each row with a helper method from Gensim. This usually takes less than 2 minutes.")
        tokens = []
        for x in sentences:
            tkns = self.tokenize(x)
            if len(tkns): 
                tokens.append(tkns)
        
        print("Saving the cleaned data set.")
        with open(settings.TOKENIZED_JOBS, 'wb') as f:
            pickle.dump(tokens, f)
        
        return tokens
    
    
    
    def try_get_model(self, dataset = None) -> Word2Vec:
        if self._model is None:       
            if(os.path.isfile(settings.W2V_MODEL)):
                print("Retrieving an existing model from "+settings.W2V_MODEL)
                self._model = Word2Vec.load(settings.W2V_MODEL)
            elif dataset is not None:
                self._model = self.train(dataset)
        return self._model
    
    
    
    def tokenize(self, sentence: str) -> list[str]:
        x = sentence
        if(isinstance(x, str)):
            x = simple_preprocess(x, deacc=True, min_len=2)
        else:
            x = []
        return x
    
    
    
    def vectorize(self, sentence: str) -> list[float] | None:
        if not self._default_vec:
            self._default_vec = [0] * self.try_get_model().wv.vector_size
        
        if isinstance(sentence, str):
            tkns = self.tokenize(sentence)
            if len(tkns) > 0:
                return self.try_get_model().wv.get_mean_vector(tkns)
            
        return self._default_vec
        


    def train(self, dataset) -> Word2Vec:
        #print("Splitting the tokenized data into training and test sets.")
        print("Training...")
        m = Word2Vec(dataset, epochs=100, vector_size=100, window=5, min_count=3, workers=os.cpu_count()-1, compute_loss=True, callbacks=[callback()])
        print("Saving the model.")
        m.save(settings.W2V_MODEL)
        
        return m
    