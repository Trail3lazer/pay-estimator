import os, settings, json
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
            
    
    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self._create_training_set()
        return self._dataset
    
    @property
    def model(self):
        if self._model is None:
            self._model = self._get_or_train()
        return self._model
    
    
    def retrain(self):
        self._dataset = self._create_training_set(overwrite=True)
        self._model = self._get_or_train(overwrite=True)
    
        
    def _create_training_set(self, overwrite=False):
        tokenized_json_path = settings.REPO_PATH +'/archive/tokenized_jobs.json'

        if(os.path.isfile(tokenized_json_path) and not overwrite):
            print("Retrieving an existing dataset at "+tokenized_json_path)
            ser = pd.read_json(tokenized_json_path)
        else:
            df = self._jobs_df.copy()
            
            bls_jobs = self._get_bls_jobs()
            
            print("Combining the the bls.gov job list, LinkedIn job title, description and skills, columns to create a single array. Word2Vec does not need them separated.")
            ser = pd.concat([bls_jobs, df['title'], df['description'], df['skills_desc']], ignore_index=True)
            
            print("Cleaning and tokenizing each row with a helper method from Gensim. This usually takes less than 2 minutes.")
            ser = ser.apply(self._prepare_sentence)
            
            print("Dropping empty rows.")
            ser.dropna(inplace=True)
            
            print("Saving the cleaned data set.")
            ser.to_json(tokenized_json_path, orient='values', indent=2)

        return ser


    def _get_or_train(self, overwrite=False):
        vectors_path = settings.REPO_PATH +'/assets/w2v/w2v.model'        
        m = None
        exists = os.path.isfile(vectors_path)
        print(exists, vectors_path)
        print(self.dataset.head())
        if(exists and not overwrite):
            print("Retrieving an existing model from "+vectors_path)
            m = Word2Vec.load(vectors_path)
        else:
            print("Splitting the tokenized data into an 80% training set and 20% test set.")
            training_set, testing_set = train_test_split(self.dataset, test_size=0.2)
            print("Training...")
            m = Word2Vec(training_set, vector_size=300, window=5, min_count=3, workers=os.cpu_count()-1)
            print("Saving the model.")
            m.save(vectors_path)
            
        return m
    
    
    def test_model(self, model, x_train, x_test):
        words = set(model.wv.index_to_key)
        X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in x_train])
        X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in x_test])
    
    
    def _prepare_sentence(self, line):
        x = line
        if(isinstance(x, str)):
            x = simple_preprocess(x, deacc=True, min_len=4)
        else:
            x = pd.NA
        return x

    #From https://www.bls.gov/ooh/a-z-index.htm
    def _get_bls_jobs(self):
        bls_jobs = json.load(open(settings.REPO_PATH +'/assets/bls_gov_jobs.json'))
        for i,x in enumerate(bls_jobs):
            joined = ' '.join(x)
            bls_jobs[i] = pd.NA if len(joined) < 4 else joined
        return pd.Series(bls_jobs).dropna()