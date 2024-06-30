import os, json, re, settings
import pandas as pd 
import numpy as np 
import pyarrow.feather as fe
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
        if(os.path.isfile(settings.CATEGORIZED_JOBS) and not overwrite):
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
    
    def get(self, path, index_col=None):
        fpath = path+'.lz4'
        df = None
        if os.path.isfile(fpath):
            df = fe.read_feather(fpath)
        elif os.path.isfile(path):
            df = pd.read_csv(path, index_col=index_col)
            fe.write_feather(df, fpath)
        return df
    

    def load_data_files(self):
        
        print("Reading CSVs")
        postings = self.get(settings.POSTINGS, index_col='job_id')
        companies = self.get(settings.COMPANIES, index_col='company_id')
        company_industries = self.get(settings.COMPANY_INDUSTRIES, index_col='company_id')
        company_specialties = self.get(settings.COMPANY_SPECIALITIES, index_col='company_id')
        company_employees = self.get(settings.EMPLOYEES, index_col='company_id')
        benefits = self.get(settings.BENEFITS, index_col='job_id')
        job_skills = self.get(settings.JOB_SKILLS, index_col='job_id')
        job_industries = self.get(settings.JOB_INDUSTRIES, index_col='job_id')
        salaries = self.get(settings.SALARIES, index_col='job_id')
        industries = self.get(settings.INDUSTRIES, index_col='industry_id')
        skills = self.get(settings.SKILLS, index_col='skill_abr')
        
        print("Joining CSV tables")
        job_industries = job_industries.join(industries, on='industry_id')
        job_skills = job_skills.join(skills, on='skill_abr')
        
        cdf = companies.join(company_industries, on='company_id')
        cdf = cdf.join(company_specialties, on='company_id')
        cdf = cdf.join(company_employees)
        
        df = postings.join(benefits, on='job_id')
        df = df.join(job_skills, on='job_id')
        df = df.join(job_industries, on='job_id')
        df = df.join(salaries, on='job_id', rsuffix='_0')
        df = df.join(cdf, on='company_id', rsuffix='_comp')
        
        print("Dropping unhelpful columns.")
        df = df.drop(axis=0,columns=[
            'company_id',
            'time_recorded',
            'original_listed_time',
            'job_posting_url',
            'application_url',
            'application_type',
            'expiry',
            'closed_time',
            'listed_time',
            'posting_domain',
            'work_type',
            'currency',
            'currency_0',
            'salary_id',
            'industry_id',
            'sponsored',
            'time_recorded',
            'remote_allowed',
            'skill_abr',
            'views',
            'applies',
            'follower_count',
            'address',
            'country',
            'city',
            'zip_code',
            'url'
        ])
        
        print("Renaming confusing columns.")
        df = df.rename(columns={
            #'name':                       'company_name',
            'type':                       'benefit_type',
            'inferred':                   'benefit_inferred',
            'formatted_experience_level': 'experience_level',
            'formatted_work_type':        'work_type',
            'title':                      'job_title',
            'description_comp':           'company_desc',
            'description':                'job_desc',
            'industry':                   'company_industry',
            'industry_name':              'job_industry'
        })
        
        print(df.info())
        
        print("Deduping columns.")
        # joining skill_name, speciality, benefit_type
        groups = df.groupby(
            by=[
                'company_name',
                'company_desc',
                'company_industry',
                'company_size',
                'employee_count',
                'job_title',
                'job_desc',
                'job_industry',
                'work_type',
                'skills_desc',
                'experience_level',
                'benefit_inferred',
                'location',
                'state',
                'experience_level',
                'work_type',
                'max_salary',
                'max_salary_0',
                'med_salary',
                'med_salary_0',
                'min_salary',
                'min_salary_0',
                'pay_period',
                'pay_period_0',
                'compensation_type',
                'compensation_type_0'],
            group_keys=False
        )
        #.apply(lambda x: ','.join(x.values))
        
        #fe.write_feather(df, settings.COMBINED_DATA)
        return df
        
    
    
    
    def _create_postings(self, overwrite=False) -> pd.DataFrame:
        if(os.path.isfile(settings.CLEANED_JOBS) and not overwrite):
            print("Retrieving an existing dataset at "+settings.CLEANED_JOBS)
            df = pd.read_pickle(settings.CLEANED_JOBS) 
            return df
        
        df = self.load_data_files()
        
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
    
    
 