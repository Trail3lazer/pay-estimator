import os
import settings
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split

class Job2Vec:
    def __init__(self, jobs_df: pd.DataFrame) -> None:
        self._dataset: pd.DataFrame = None
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
        return self._dataset
    
    
    def retrain(self):
        self._dataset = self._create_training_set(overwrite=True)
        self._model = self._get_or_train(overwrite=True)
    
        
    def _create_training_set(self, overwrite=False):
        df = self._jobs_df.copy()
        tokenized_csv_path = settings.REPO_PATH +'/archive/tokenized_jobs.csv'

        if(os.path.isfile(tokenized_csv_path) and not overwrite):
            print("Retrieving an existing dataset at "+tokenized_csv_path)
            ser = pd.read_csv(tokenized_csv_path, index_col=0) 
        else:
            print("Combining the job title and description columns to create a single array. Word2Vec does not need them separated.")
            ser = pd.concat([df['title'], df['description']], ignore_index=True)
            
            print("Cleaning and tokenizing each row with a helper method from Gensim. This usually takes a little more than a minute.")
            ser = ser.apply(self._prepare_sentence)
            
            print("Dropping empty rows.")
            ser.dropna(inplace=True)
            
            print("Saving the training and test sets as a csv file.")
            ser.to_csv(tokenized_csv_path)

        return ser


    def _get_or_train(self, overwrite=True):
        vectors_path = settings.REPO_PATH +'/assets/word_vectors.model'
        model = None

        if(os.path.isfile(vectors_path) and not overwrite):
            model = Word2Vec.load(vectors_path)
        else:
            print("Splitting the tokenized data into an 80% training set and 20% test set.")
            x_train, x_test = train_test_split(self.dataset, test_size=0.2)
            model = Word2Vec(x_train, vector_size=100, window=3, min_count=10, workers=8)
            model.wv.save(vectors_path)
        return model
    
    def test_model(self, model, x_train, x_test):
        words = set(model.wv.index_to_key )
        X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in x_train])
        X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in x_test])
    
    def _prepare_sentence(self, line):
        if(isinstance(line, str)):
            line = simple_preprocess(line)
        else:
            line = pd.NA
        return line
