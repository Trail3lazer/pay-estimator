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
        self._default_vec: list[float] = None
    
    
    
    def get_dataset(self, jobs_df: pd.DataFrame = None,) -> pd.Series:
        if self._dataset is None:
            if(os.path.isfile(settings.TOKENIZED_JOBS)):
                print("Retrieving an existing dataset at "+settings.TOKENIZED_JOBS)
                self._dataset = pd.read_pickle(settings.TOKENIZED_JOBS)
            elif isinstance(jobs_df, pd.DataFrame): 
                self._jobs_df = jobs_df
                
            if self._jobs_df is None: 
                raise ValueError("Job data must be passed in before the data can be prepared for training.")
            
            self._dataset = self.prepare_training_data()
        return self._dataset
    
    
    
    def get_model(self) -> Word2Vec:
        if self._model is None:       
            if(os.path.isfile(settings.W2V_MODEL)):
                print("Retrieving an existing model from "+settings.W2V_MODEL)
                self._model = Word2Vec.load(settings.W2V_MODEL)
            else:
                self._model = self.train()
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
        
        
        
    def prepare_training_data(self, overwrite=False) -> pd.Series:
        df = self._jobs_df.copy()
        
        bls_jobs = self._get_bls_jobs()
        
        print("Combining the the bls.gov job list, LinkedIn job title, description and skills, columns to create a single array. Word2Vec does not need them separated.")
        ser = pd.concat([bls_jobs, df['title'], df['description'], df['skills_desc']], ignore_index=True)
        
        print("Cleaning and tokenizing each row with a helper method from Gensim. This usually takes less than 2 minutes.")
        ser = ser.apply(self.tokenize)
        
        print("Dropping empty rows.")
        ser.dropna(inplace=True)
        
        print("Saving the cleaned data set.")
        ser.to_pickle(settings.TOKENIZED_JOBS)

        return ser



    def train(self, overwrite=False) -> Word2Vec:
        print("Splitting the tokenized data into training and test sets.")
        training_set, testing_set = train_test_split(self.get_dataset(), test_size=0.05)
        print("Training...")
        m = Word2Vec(training_set, vector_size=300, window=5, min_count=3, workers=os.cpu_count()-1)
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
    
    
    
    #From https://www.bls.gov/ooh/a-z-index.htm
    def _get_bls_jobs(self) -> pd.Series:
        bls_jobs = json.load(open(settings.BLS_JOBS))
        for i,x in enumerate(bls_jobs):
            joined = ' '.join(x)
            bls_jobs[i] = pd.NA if len(joined) < 4 else joined
        return pd.Series(bls_jobs).dropna()