import os, json, settings
import pandas as pd
import numpy as np
import dask.dataframe as dd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split

class Job2Vec:
    def __init__(self):
        self._jobs_df: pd.DataFrame = None
        self._dataset: pd.Series = None
        self._model = None
        self._default_vec: list[float] = None
    
    
    
    def preprocess_data(self, df: pd.DataFrame = None, bls_jobs: pd.Series = None) -> pd.Series:

        if(os.path.isfile(settings.TOKENIZED_JOBS)):
            print("Retrieving an existing dataset at "+settings.TOKENIZED_JOBS)
            return dd.read_parquet(settings.TOKENIZED_JOBS)
                
        print("Combining the the bls.gov job list, LinkedIn job title, description and skills, columns to create a single array. Word2Vec does not need them separated.")
        data = [bls_jobs, df['job_title'].unique(), df['job_desc'].unique(), df['skills_desc'].unique(), df['company_desc'].unique(), df['company_industry'].unique()]
        ser = pd.concat(data, ignore_index=True)
        
        print("Cleaning and tokenizing each row with a helper method from Gensim. This usually takes less than 2 minutes.")
        ser: pd.Series = ser.apply(self.tokenize)
        
        print("Dropping empty rows.")
        ser.dropna(inplace=True)
        
        print("Saving the cleaned data set.")
        ser.to_parquet(settings.TOKENIZED_JOBS)

        return ser
    
    
    
    def get_model(self, dataset = None) -> Word2Vec:
        if self._model is None:       
            if(os.path.isfile(settings.W2V_MODEL)):
                print("Retrieving an existing model from "+settings.W2V_MODEL)
                self._model = Word2Vec.load(settings.W2V_MODEL)
            else:
                self._model = self.train(dataset)
        return self._model
    
    
    
    def retrain(self) -> None:
        self._dataset = self.prepare_training_data(overwrite=True)
        self._model = self.train(overwrite=True)
    
    
    
    def tokenize(self, sentence: str) -> list[str]:
        x = sentence
        if(isinstance(x, str)):
            x = simple_preprocess(x, deacc=True, min_len=2)
        else:
            x = []
        return x
    
    
    
    def vectorize(self, sentence: str) -> list[float] | None:
        if not self._default_vec:
            self._default_vec = [0] * self.get_vector_length()
        
        if isinstance(sentence, str):
            tkns = self.tokenize(sentence)
            if len(tkns) > 0:
                return self.get_model().wv.get_mean_vector(tkns)
            
        return self._default_vec
    
    
        
    def get_vector_length(self):
        return self.get_model().wv.vector_size
        


    def train(self, dataset) -> Word2Vec:
        print("Splitting the tokenized data into training and test sets.")
        training_set, testing_set = train_test_split(dataset, test_size=0.05)
        print("Training...")
        m = Word2Vec(training_set, vector_size=100, window=5, min_count=3, workers=os.cpu_count()-1)
        print("Saving the model.")
        m.save(self.model_path)
        
        self.test_model(m, training_set, testing_set)
        
        return m
    
    
    
    def test_model(self, model, x_train, x_test):
        words = set(model.wv.index_to_key)
        X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in x_train])
        X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in x_test])
        print(X_train_vect)
        print(X_test_vect)