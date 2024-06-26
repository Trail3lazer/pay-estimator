import os, json, re, settings
import pandas as pd 
import numpy as np 
from scipy import stats 
from typing import Callable

class DataManager:
        
    def __init__(self) -> None:
        self._backup_postings: pd.DataFrame = None
        self._postings: pd.DataFrame = None
        self._postings_with_pay: pd.DataFrame = None
        self._state_abbr: dict[str,str] = None
        self._pay_cols = ['max_salary','med_salary','min_salary']
        
        # https://www.bls.gov/charts/american-time-use/emp-by-ftpt-job-edu-h.htm
        self._fulltime = self._calculate_work_week_hours(8.42, 0.874, 5.57, 0.287)
        self._parttime = self._calculate_work_week_hours(5.54, 0.573, 5.48, 0.319)

        # https://www.bls.gov/ebs/factsheets/paid-vacations.htm#:~:text=The%20number%20of%20vacation%20days,19%20days%20of%20paid%20vacation.
        vacation_day_pcts = np.matrix([  
        #<5, <10, <15, <20, <25, >24
        [8,	31,	34,	18,	7,	2],
        [3,	12,	30,	32,	16,	7],
        [2,	8,	18,	33,	23,	17],
        [2,	8,	14,	20,	29,	28]
        ])
    
        # https://www.ca2.uscourts.gov/clerk/calendars/federal_holidays.html
        holidays = 11 
        weeks_off = (holidays + self._calculate_vacation_days(vacation_day_pcts))/7
        self._weeks = 52
        self._work_weeks = self._weeks - weeks_off
        self._months = 12
        
        
    
    def get_postings(self):
        if self._postings is None:
            if self._backup_postings is None:
                self._backup_postings = self._create_postings()
            self._postings = self._backup_postings.copy()
        return self._postings

    
    
    def get_postings_with_pay(self):
        if self._postings_with_pay is None:
            self._postings_with_pay = self._drop_jobs_missing_pay()
        return self._postings_with_pay



    def reset_postings(self):
        self._backup_postings = None
        self._postings = None
        self._salary_postings = None
    
    
    
    def categorize_job_titles(self, get_similar_categories: Callable[[pd.Series, int],list[tuple[str,float]]], overwrite = False):
        if(os.path.isfile(settings.Ca) and not overwrite):
            print("Retrieving an existing data at "+settings.CATEGORIZED_JOBS)
            df = pd.read_pickle(settings.CATEGORIZED_JOBS)
            return df
        
        df = self.get_postings()
        category_count = 3
        for i in range(category_count):
            df[f'cat{i}'] = None
            df[f'cat{i}_score'] = 0

        def apply_categories(row):
            try:
                categories = get_similar_categories(row['title'], category_count)
            except:
                categories = [('',0),('',0),('',0)]
            for i in range(category_count):
                category = categories[i]
                if isinstance(category, tuple):
                    row[f'cat{i}'] = str(category)
                row[f'cat{i}'] = categories[i][0]
                row[f'cat{i}_score'] = categories[i][1]
            return row

        df: pd.DataFrame = df.apply(apply_categories, axis=1)

        df.to_pickle(settings.CATEGORIZED_JOBS)
        return df
    
    
    
    def _create_postings(self, overwrite=False) -> pd.DataFrame:
        if(os.path.isfile(settings.CLEANED_JOBS) and not overwrite):
            print("Retrieving an existing dataset at "+settings.CLEANED_JOBS)
            df = pd.read_pickle(settings.CLEANED_JOBS) 
            return df
        
        print("Reading CSV")
        df = pd.read_csv(settings.ORIGINAL_CSV)
        
        columns_to_drop = [
            'job_id','company_id','formatted_work_type','currency','views','applies',
            'original_listed_time','remote_allowed','job_posting_url','application_url',
            'application_type','expiry','closed_time','listed_time','posting_domain',
            'sponsored','compensation_type','sponsored',
            ]
        
        print("Dropping unhelpful columns: "+str(columns_to_drop))
        df.drop(columns_to_drop, axis=1, inplace=True)
        
        print("Reading the state abbreviation json map.")
        if(self._state_abbr is None): 
            self._state_abbr = dict(json.load(open(settings.STATE_ABBR)))
            
        print("Creating a state abbreviation column from the location column and normalizing the pay columns.")
        df['state'] = ''
        df: pd.DataFrame = df.apply(self._normalize_row, axis=1)
        
        print("Setting outlier pay column values to NaN.")
        for name in self._pay_cols:
            zscore_thresh = 5 if name == 'med_salary' else 3
            mask = (np.abs(stats.zscore(df[name].astype(float), nan_policy='omit')) > zscore_thresh) | (df[name].astype(float) < 10000)
            df[name] = df[name].mask(mask, np.NaN)
            
        print("Creating an average salary column that is is the average of the salary pay columns. "+str(self._pay_cols))
        df['avg_salary'] = df[self._pay_cols].mean(axis=1)
        
        print('Saving cleaned the posting table so we do not need to process it each time.')
        df.to_pickle(settings.CLEANED_JOBS)
        return df
    
    

    def _update_pay(self, row, mult):
        for c in self._pay_cols:
            if row[c] != row[c]:
                continue
            row[c] = row[c] * mult



    def _clean_pay(self, row):
        pay_period = row['pay_period']
        if pay_period == 'MONTHLY': 
            self._update_pay(row, self._months)
        elif pay_period == 'WEEKLY': 
            self._update_pay(row, self._weeks)
        elif pay_period == 'HOURLY':
            hours = (self._parttime if row['work_type'] == 'PART_TIME' else self._fulltime) * self._work_weeks
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
                    row[c] = row[c] * self._weeks / 2
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
        return self.get_postings().copy().dropna(thresh=1, subset=self._pay_cols)
        
        
        
    def get_abnormal_states(self, ser):
        return ser[ser.str.len() != 2].unique()
    
    
    
    def _calculate_work_week_hours(self, weekday_hrs, week_day_pct, weekend_hrs, weekend_pct):
        total_weekday = weekday_hrs * week_day_pct * 5
        total_weekend = weekend_hrs * weekend_pct * 2
        return total_weekday + total_weekend
    
    
    
    def _calculate_vacation_days(self, pcts: np.matrix):
        days = []
        it = np.nditer(pcts, flags=['c_index','multi_index'])
        for pct in it:
            i, j = it.multi_index
            days.append(np.sum([day*pct/100 for day in range((j*5),(j*5)+5)]))
        avg = np.mean(days)
        return avg
    
    
 