import os, json, settings
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split

class Job2Vec:
    def __init__(self):
        self._jobs_df: pd.DataFrame = None
        self._dataset: pd.Series = None
        self._model = None
    
    
    
    def get_dataset(self, jobs_df: pd.DataFrame = None,) -> pd.Series:
        if self._dataset is None:
            if(os.path.isfile(settings.TOKENIZED_JOBS)):
                print("Retrieving an existing dataset at "+settings.TOKENIZED_JOBS)
                self._dataset = pd.read_pickle(settings.TOKENIZED_JOBS)
            elif isinstance(jobs_df, pd.DataFrame): 
                self._jobs_df = jobs_df
            if not self._jobs_df: 
                raise ValueError("Job data must be passed in before the data can be prepared for training.")
            self._dataset = self._create_training_set()
        return self._dataset
    
    
    
    def get_model(self) -> Word2Vec:
        if self._model is None:       
            if(os.path.isfile(settings.W2V_MODEL)):
                print("Retrieving an existing model from "+settings.W2V_MODEL)
                self._model = Word2Vec.load(settings.W2V_MODEL)
            else:
                self._model = self._get_or_train()
        return self._model
    
    
    
    def retrain(self) -> None:
        self._dataset = self._create_training_set(overwrite=True)
        self._model = self._get_or_train(overwrite=True)
    
    
    
    def tokenize(self, sentence: str) -> list[str]:
        x = sentence
        if(isinstance(x, str)):
            x = simple_preprocess(x, deacc=True, min_len=2)
        else:
            x = []
        return x
    
    
        
    def _create_training_set(self, overwrite=False) -> pd.Series:
        df = self._jobs_df.copy()
        
        bls_jobs = self._get_bls_jobs()
        
        print("Combining the the bls.gov job list, LinkedIn job title, description and skills, columns to create a single array. Word2Vec does not need them separated.")
        ser = pd.concat([bls_jobs, df['title'], df['description'], df['skills_desc']], ignore_index=True)
        
        print("Cleaning and tokenizing each row with a helper method from Gensim. This usually takes less than 2 minutes.")
        ser = ser.apply(self.tokenize)
        
        print("Dropping empty rows.")
        ser.dropna(inplace=True)
        
        print("Saving the cleaned data set.")
        ser.to_pickle(self.tokenized_data_path)

        return ser



    def _get_or_train(self, overwrite=False) -> Word2Vec:
        print("Splitting the tokenized data into an 80% training set and 20% test set.")
        training_set, testing_set = train_test_split(self.get_dataset(), test_size=0.2)
        print("Training...")
        m = Word2Vec(training_set, vector_size=300, window=5, min_count=3, workers=os.cpu_count()-1)
        print("Saving the model.")
        m.save(self.model_path)
            
        return m
    
    
    
    def test_model(self, model, x_train, x_test):
        words = set(model.wv.index_to_key)
        X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in x_train])
        X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in x_test])
    
    
    
    #From https://www.bls.gov/ooh/a-z-index.htm
    def _get_bls_jobs(self) -> pd.Series:
        bls_jobs = json.load(open(settings.BLS_JOBS))
        for i,x in enumerate(bls_jobs):
            joined = ' '.join(x)
            bls_jobs[i] = pd.NA if len(joined) < 4 else joined
        return pd.Series(bls_jobs).dropna()