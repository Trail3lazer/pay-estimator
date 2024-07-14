print('Importing libraries for the DataManager class.')
import os, json, re, settings
from numbers import Number
import pandas as pd 
import numpy as np
from scipy import stats 
from typing import Callable

class DataManager:
        
    def __init__(self) -> None:
        self._backup_postings: pd.DataFrame = None
        self._postings: pd.DataFrame = None
        self._postings_with_pay: pd.DataFrame = None
        self._state_abbr: dict[str,str] = dict(json.load(open(settings.STATE_ABBR)))
        self._state_re: re.Pattern = self._build_state_match_re()
        self._pay_cols = ['max_salary','med_salary','min_salary']
        self._bckt_size = 1
        self._pay_table_suffix = '_from_salaries'
        
        # https://www.bls.gov/charts/american-time-use/emp-by-ftpt-job-edu-h.htm
        self._fulltime = self._calculate_work_week_hours(8.42, 0.874, 5.57, 0.287)
        self._parttime = self._calculate_work_week_hours(5.54, 0.573, 5.48, 0.319)

        # https://www.bls.gov/ebs/factsheets/paid-vacations.htm#:~:text=The%20number%20of%20vacation%20days,19%20days%20of%20paid%20vacation.
        vacation_day_pcts = np.array([  
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
                if(os.path.isfile(settings.CLEANED_JOBS)):
                    print("Retrieving an existing dataset at "+settings.CLEANED_JOBS)
                    self._backup_postings = pd.read_parquet(settings.CLEANED_JOBS)
                else:
                    raw = self.read_postings()
                    self._backup_postings = self._create_postings(raw)
                    
                    print('Saving cleaned the posting table so we do not need to process it each time.')
                    self._backup_postings.to_parquet(settings.CLEANED_JOBS)
            self._postings = self._backup_postings.copy()
        return self._postings



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
    
    
    
    def read(self, path: str, index_col=None):
        fpath = path.replace(settings.ARCHIVE_EXT,'.pqt')
        df: pd.DataFrame = None
        if os.path.isfile(fpath):
            df = pd.read_parquet(fpath)
        elif os.path.isfile(path):
            try:
                df = pd.read_csv(path, on_bad_lines='warn',).set_index(index_col)
            except (Exception) as detail: 
                print(path, detail)
            df.to_parquet(fpath)
            os.remove(path)
        return df
    
    
    
    def load_additional_tables(self):
        industries = self.read(settings.INDUSTRIES, index_col='industry_id')['industry_name'].to_numpy()
        skills = self.read(settings.SKILLS, index_col='skill_abr')['skill_name'].to_numpy()
        benefits = self.read(settings.BENEFITS, index_col='job_id')['type'].unique()
        company_industries = self.read(settings.COMPANY_INDUSTRIES, index_col='company_id')['industry'].unique()
        company_specialities = self.read(settings.COMPANY_SPECIALITIES, index_col='company_id')['speciality'].unique()
        
        return [benefits,skills,industries,company_industries,company_specialities]
        
    
    
    def get_bls_jobs(self) -> pd.Series:
        bls_jobs = json.load(open(settings.BLS_JOBS))
        for i,x in enumerate(bls_jobs):
            joined = ' '.join(x)
            bls_jobs[i] = pd.NA if len(joined) < 4 else joined
        return pd.Series(bls_jobs).dropna()
    
    
    
    def get_bls_jobs(self) -> pd.Series:
        bls_jobs = json.load(open(settings.BLS_JOBS))
        for i,x in enumerate(bls_jobs):
            joined = ' '.join(x)
            bls_jobs[i] = pd.NA if len(joined) < 4 else joined
        return pd.Series(bls_jobs).dropna()
    
    
    
    def read_postings(self):        
        print("Reading tables")
        postings = self.read(settings.POSTINGS, index_col='job_id')
        companies = self.read(settings.COMPANIES, index_col='company_id')
        company_employees = self.read(settings.EMPLOYEES, index_col='company_id')
        salaries = self.read(settings.SALARIES, index_col='job_id')
        
        print("Joining tables")
        df = postings.join(salaries, on='job_id', rsuffix=self._pay_table_suffix)
        cdf = companies.join(company_employees, on='company_id')
        df = df.join(cdf, on='company_id', rsuffix='_comp')
        
        return df
    
    
    
    def _create_postings(self, df) -> pd.DataFrame:
        
        print("Dropping unhelpful columns.")
        df = df.drop(axis=0, columns=[
            'company_id',
            'time_recorded',
            'original_listed_time',
            'job_posting_url',
            'application_url',
            'application_type',
            'expiry',
            'closed_time',
            'posting_domain',
            'work_type',
            'currency',
            'currency'+self._pay_table_suffix,
            'salary_id',
            'sponsored',
            'time_recorded',
            'remote_allowed',
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
            'type':                       'benefit_type',
            'inferred':                   'benefit_inferred',
            'formatted_experience_level': 'experience_level',
            'formatted_work_type':        'work_type',
            'title':                      'job_title',
            'description_comp':           'company_desc',
            'description':                'job_desc',
            'industry':                   'company_industry',       
        })
        
        print("Creating a state abbreviation column from the location column and normalizing the pay columns.")
        df = df.apply(self._normalize_row, axis=1)
        
        print("Setting outlier pay column values to NaN.")
        for name in self._pay_cols:
            zscore_thresh = 5 if name == 'med_salary' else 3
            mask = (np.abs(stats.zscore(df[name].astype(float), nan_policy='omit')) > zscore_thresh) | (df[name].astype(float) < 10000)
            df[name] = df[name].mask(mask, np.NaN)
            
        print("Creating an average salary column that is is the average of the salary pay columns. "+str(self._pay_cols))
        mean = df[self._pay_cols].mean(axis=1).to_numpy()
        quotient = mean / self._bckt_size
        rounded = np.ceil(quotient) 
        df['pay'] = rounded * self._bckt_size
        
        #dup_cols = ['job_title','company_name','job_desc','state','pay','listed_time']
        #print(f"Dropping duplicate jobs based on these colums: {', '.join(dup_cols)}.")
        #df: pd.DataFrame = df.drop_duplicates(subset=dup_cols, ignore_index=True)
        
        return df
    
    
    
    def get_or_create_categorized_postings(self, categorize_func):
        if os.path.isfile(settings.CATEGORIZED_JOBS):
            print(f'Retrieving categorized jobs from file {settings.CATEGORIZED_JOBS}')
            df = pd.read_parquet(settings.CATEGORIZED_JOBS)
        else:
            df = self.get_postings()
            df = df[['job_title','pay','state']].copy()

            df['category'] = None

            def categorize(row):
                category = categorize_func(row['job_title'])
                if isinstance(category, tuple):
                    row['category'] = category[0]
                return row

            print('Categorizing jobs.')
            df: pd.DataFrame = df.apply(categorize, axis=1)
            
            df.to_parquet(settings.CATEGORIZED_JOBS)
        return df
    
    
    
    def update_pay(self, row, mult):
        for c in self._pay_cols:
            if isinstance(row[c], Number) and row[c] > 0:
                row[c] = round(row[c] * mult,2)
            elif isinstance(row[c+self._pay_table_suffix], Number):
                row[c] = round(row[c+self._pay_table_suffix] * mult,2)
        return row



    def clean_pay(self, row):
        pay_period:str = row['pay_period'] or row['pay_period'+self._pay_table_suffix]
        if(not isinstance(pay_period, str)):
            return row
        if pay_period == 'MONTHLY': 
            row = self.update_pay(row, self._months)
        elif pay_period == 'WEEKLY': 
            row = self.update_pay(row, self._weeks)
        elif pay_period == 'HOURLY':
            work_type = row['work_type']
            if isinstance(work_type, str) and work_type == 'PART_TIME':
                hours = self._parttime * self._work_weeks
            else: 
                hours = self._fulltime * self._work_weeks
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
        else:
            row = self.update_pay(row, 1)
        return row



    def salary_to_hourly(self, salary: float, pay_period: str, work_type='FULL_TIME'):
        if(not isinstance(pay_period, str)):
            return salary
        if pay_period == 'MONTHLY': 
            return salary / self._months
        elif pay_period == 'WEEKLY': 
            return salary / self._weeks
        elif pay_period == 'HOURLY':
            if isinstance(work_type, str) and work_type == 'PART_TIME':
                hours = self._parttime * self._work_weeks
            else: 
                hours = self._fulltime * self._work_weeks
            return salary/hours
        elif pay_period == 'BIWEEKLY':
            return salary / self._weeks * 2
        return salary



    def try_get_state_abbr(self, location):
        location = self._clean_loc_str(location)
        
        name_match = self._state_re.search(location)
        if name_match is None:
            return location  
        
        name_match = name_match.group()
        if len(name_match) == 2:
            return name_match
    
        return self._state_abbr.get(name_match)



    def _clean_loc_str(self, loc: str):
        if not isinstance(loc, str):
            return loc
        loc = loc.strip().upper()
        loc = loc.replace('.','')
        return loc
    
    
    
    def _is_valid_state(self, state):
        return isinstance(state, str) and len(state) == 2 and state in self._state_abbr.values()
    


    def _clean_state(self, row):
        hasLocationData = isinstance(row['location'], str) or isinstance(row['state'], str)
        if not isinstance(row, pd.Series) or not hasLocationData: 
            return row
            
        state = ''
        
        if isinstance(row['location'], str):

            location = row['location'].split(',')
            
            if len(location) > 1:
                state = self._clean_loc_str(location[1])
            else:
                state = self._clean_loc_str(location[0])

            if len(state) != 2:
                state = self.try_get_state_abbr(row['location'])
        
        if not self._is_valid_state(state) and isinstance(row['state'], str):
            
            state = self._clean_loc_str(row['state'])
            
            if not self._is_valid_state(state):
                state = self.try_get_state_abbr(state)
        
        row['state'] = state if self._is_valid_state(state) else None

        return row



    def _normalize_row(self, row):
        row = self._clean_state(row)
        row = self.clean_pay(row)
        return row
        
        
        
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
    
    
    
    def _build_state_match_re(self):
        return re.compile(r'\b(?:Alabama|AL|Alaska|AK|Arizona|AZ|Arkansas|AR|California|CA|Colorado|CO|Connecticut|CT|Delaware|DE|Florida|FL|Georgia|GA|Hawaii|HI|Idaho|ID|Illinois|IL|Indiana|IN|Iowa|IA|Kansas|KS|Kentucky|KY|Louisiana|LA|Maine|ME|Maryland|MD|Massachusetts|MA|Michigan|MI|Minnesota|MN|Mississippi|MS|Missouri|MO|Montana|MT|Nevada|NV|New\s+Hampshire|NH|New\s+Jersey|NJ|New\s+Mexico|NM|New\s+York|NY|North\s+Carolina|NC|North\s+Dakota|ND|Ohio|OH|Oklahoma|OK|Oregon|OR|Pennsylvania|PA|Rhode\s+Island|RI|South\s+Carolina|SC|South\s+Dakota|SD|Tennessee|TN|Texas|TX|Utah|UT|Vermont|VT|Virginia|VA|Washington\s+DC|DC|Washington|WA|West\s+Virginia|WV|Wisconsin|WI|Wyoming|WY|Nebraska|NE)\b', re.IGNORECASE)