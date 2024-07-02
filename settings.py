import os

REPO_PATH = os.path.abspath('.')

JOB_CATEGORIES = REPO_PATH + '/assets/job_categories.json'
JOB_FIELDS = REPO_PATH + '/assets/job_fields.json'
BLS_JOBS = REPO_PATH +'/assets/bls_gov_jobs.json' #From https://www.bls.gov/ooh/a-z-index.htm
STATE_ABBR = REPO_PATH + '/assets/state_abbr.json' #This file was modified from this github gist 

POSTINGS = REPO_PATH + '/archive/postings.csv'
COMPANIES = REPO_PATH + '/archive/companies/companies.csv'
COMPANY_INDUSTRIES = REPO_PATH + '/archive/companies/company_industries.csv'
COMPANY_SPECIALITIES = REPO_PATH + '/archive/companies/company_specialities.csv'
EMPLOYEES = REPO_PATH + '/archive/companies/employee_counts.csv'
BENEFITS = REPO_PATH + '/archive/jobs/benefits.csv'
JOB_SKILLS = REPO_PATH + '/archive/jobs/job_skills.csv'
JOB_INDUSTRIES = REPO_PATH + '/archive/jobs/job_industries.csv'
SALARIES = REPO_PATH + '/archive/jobs/salaries.csv'
SKILLS = REPO_PATH + '/archive/mappings/skills.csv'
INDUSTRIES = REPO_PATH + '/archive/mappings/industries.csv'

COMBINED_DATA = REPO_PATH + '/archive/app/data.lz4'

CLEANED_JOBS = REPO_PATH + '/archive/app/clean_postings.pqt'

TOKENIZED_JOBS = REPO_PATH +'/archive/app/tokenized_jobs.pqt'
W2V_MODEL = REPO_PATH +'/assets/w2v/w2v.model'

CATEGORY_VECS = REPO_PATH + '/assets/w2v/vectorized_categories.bin'

CATEGORIZED_JOBS = REPO_PATH + '/archive/app/categorized_job_titles.pqt' 

ENTEMB_MODEL = REPO_PATH + 'assets/entemb/entemb.model'