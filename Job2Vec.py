import os, settings, json, pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split

class Job2Vec:
    def __init__(self, jobs_df: pd.DataFrame) -> None:
        self._dataset: np.array = None
        self._model = None
        self._jobs_df = jobs_df
        self.tokenized_data_path = settings.REPO_PATH +'/archive/tokenized_jobs.bin'
        self.model_path = settings.REPO_PATH +'/assets/w2v/w2v.model'  
            
    
    def get_dataset(self):
        if self._dataset is None:
            if(os.path.isfile(self.tokenized_data_path)):
                print("Retrieving an existing dataset at "+self.tokenized_data_path)
                self._dataset = pd.read_pickle(self.tokenized_data_path)
            else:
                self._dataset = self._create_training_set()
        return self._dataset
    
    
    def get_model(self):
        if self._model is None:       
            if(os.path.isfile(self.model_path)):
                print("Retrieving an existing model from "+self.model_path)
                self._model = Word2Vec.load(self.model_path)
            else:
                self._model = self._get_or_train()
        return self._model
    
    
    def retrain(self):
        self._dataset = self._create_training_set(overwrite=True)
        self._model = self._get_or_train(overwrite=True)
    
    
    def tokenize(self, sentence):
        x = sentence
        if(isinstance(x, str)):
            x = simple_preprocess(x, deacc=True, min_len=4)
        else:
            x = []
        return x
    
        
    def _create_training_set(self, overwrite=False):

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


    def _get_or_train(self, overwrite=False):
        
        print("Splitting the tokenized data into an 80% training set and 20% test set.")
        training_set, testing_set = train_test_split(self.get_dataset, test_size=0.2)
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
    def _get_bls_jobs(self):
        bls_jobs = json.load(open(settings.REPO_PATH +'/assets/bls_gov_jobs.json'))
        for i,x in enumerate(bls_jobs):
            joined = ' '.join(x)
            bls_jobs[i] = pd.NA if len(joined) < 4 else joined
        return pd.Series(bls_jobs).dropna()