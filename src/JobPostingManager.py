import os
import json
import re
import settings
import pandas as pd
import numpy as np
from scipy import stats

class JobPostingManager:
    def __init__(self) -> None:
        self._backup_postings: pd.DataFrame = None
        self._postings: pd.DataFrame = None
        self._salary_postings: pd.DataFrame = None
        self._state_abbr: dict[str,str] = None

    @property
    def postings(self):
        if self._postings is None:
            if self._backup_postings is None:
                self._backup_postings = self._create_postings()
            self._postings = self._backup_postings.copy()
        return self._postings

    @property
    def salary_postings(self):
        if self._salary_postings is None:
            self._salary_postings = self._create_salary_postings()
        return self._salary_postings

    def reset_postings(self):
        self._backup_postings = None
        self._postings = None
        self._salary_postings = None
    
    def _clean_state(self,row):    
        if row['location'] != row['location']: 
            return row

        location = row['location'].strip().split(',')

        if len(location) == 0:
            return row

        state = ''

        if len(location) > 1:
            state = location[1].strip().upper()

        if len(state) != 2:
            for k in self._state_abbr.keys():
                result = re.search(k, row['location'], flags=re.I)
                if result is not None:
                    state = self._state_abbr.get(k)
                    break
                
        if state != None and len(state) == 2:
            row['state'] = state
        else:
            row['state'] = None

        return row


    def _create_postings(self):
        print("Reading CSV")
        
        df = pd.read_csv(settings.REPO_PATH + '/archive/postings.csv')

        columns_to_drop = [
            'views','applies','original_listed_time','remote_allowed','job_posting_url','application_url','application_type',
            'expiry','closed_time','listed_time','posting_domain','sponsored','compensation_type','sponsored',
            ]
        
        print("Dropping unhelpful columns: "+str(columns_to_drop))
        
        df.drop(columns_to_drop, axis=1, inplace=True)
        
        print("Reading the state abbreviation json map.")
        
        if(self._state_abbr is None): 
            self._state_abbr = dict(json.load(open(settings.REPO_PATH + '/assets/state_abbr.json')))

        print("Creating a state abbreviation column from the location column.")
        
        df['state'] = ''
        df = df.apply(self._clean_state, axis=1)

        print(df.head())
        return df
    
    
    def _create_salary_postings(self):
        df = self.postings.copy()
        pay_cols = ['max_salary','med_salary','min_salary']
        
        print("Dropping rows where every salary column is empty or the listing is an hourly job.")
        
        df = df.loc[df['pay_period'] != 'HOURLY'].copy().dropna(thresh=1, subset=pay_cols)
        
        print("Setting outlier salary column values to NaN where the zscore is greater than 2.")
        
        for name in pay_cols:
            mask = np.abs(stats.zscore(df[name].astype(float), nan_policy='omit')) > 2
            df[name] = df[name].mask(mask, np.NaN)
        
        print("Creating an average salary column that is is the average of the salary pay columns. "+str(pay_cols))
        
        df['avg_salary'] = df[pay_cols].mean(axis=1)
        
        return df
    
    
    def get_abnormal_states(self, ser):
        return ser[ser.str.len() != 2].unique()
    
 