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
        self._postings_with_pay: pd.DataFrame = None
        self._state_abbr: dict[str,str] = None
        self._pay_cols = ['max_salary','med_salary','min_salary']

    @property
    def postings(self):
        if self._postings is None:
            if self._backup_postings is None:
                self._backup_postings = self._create_postings()
            self._postings = self._backup_postings.copy()
        return self._postings

    @property
    def postings_with_pay(self):
        if self._postings_with_pay is None:
            self._postings_with_pay = self._drop_jobs_missing_pay()
        return self._postings_with_pay

    def reset_postings(self):
        self._backup_postings = None
        self._postings = None
        self._salary_postings = None
    
    
    def _create_postings(self, overwrite=False):
        cached = settings.REPO_PATH + '/archive/clean_postings.csv'

        if(os.path.isfile(cached) and not overwrite):
            print("Retrieving an existing dataset at "+cached)
            df = pd.read_csv(cached, index_col=0) 
        else:
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
                
            print("Creating a state abbreviation column from the location column and normalizing the pay columns.")
            df['state'] = ''
            df = df.apply(self._normalize_row, axis=1)
            
            print("Setting outlier pay column values to NaN.")
            for name in self._pay_cols:
                zscore_thresh = 5 if name == 'med_salary' else 3
                mask = (np.abs(stats.zscore(df[name].astype(float), nan_policy='omit')) > zscore_thresh) | (df[name].astype(float) < 10000)
                df[name] = df[name].mask(mask, np.NaN)
                
            print("Creating an average salary column that is is the average of the salary pay columns. "+str(self._pay_cols))
            df['avg_salary'] = df[self._pay_cols].mean(axis=1)
            
            print('Saving cleaned the posting table so we do not need to process it each time.')
            df.to_csv(cached)
        return df
    

    def _update_pay(self, row, mult):
        for c in self._pay_cols:
            if row[c] != row[c]:
                continue
            row[c] = row[c] * mult


    def _clean_pay(self, row):
        # Keep YEARLY
        # Monthly * 12
        # WEEKLY * 52
        # HOURLY * (Part-time ? 20 : 40) * 52
        # BIWEEKLY drop lt 10000
        pay_period = row['pay_period']
        if pay_period == 'MONTHLY': 
            self._update_pay(row, 12)
        elif pay_period == 'WEEKLY': 
            self._update_pay(row, 52)
        elif pay_period == 'HOURLY':
            hours = (20 if row['work_type'] == 'PART_TIME' else 40) * 52
            for c in self._pay_cols:
                if row[c] != row[c]:
                    continue
                if row[c] < 1000:
                    row[c] = row[c] * hours
        elif pay_period == 'BIWEEKLY':
            for c in self._pay_cols:
                if row[c] != row[c]:
                    continue
                if row[c] < 10000:
                    row[c] = row[c] * 26
        return row


    def _clean_state(self, row):    
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


    def _normalize_row(self, row):
        row = self._clean_state(row)
        row = self._clean_pay(row)
        return row


    def _drop_jobs_missing_pay(self):
        print("Dropping rows where every pay column is empty.")
        return self.postings.copy().dropna(thresh=1, subset=self._pay_cols)
        
        
    def get_abnormal_states(self, ser):
        return ser[ser.str.len() != 2].unique()
    
 