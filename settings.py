import os

REPO_PATH = os.path.abspath('.')

# Dataset
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

# Additional data
JOB_CATEGORIES = REPO_PATH + '/assets/job_categories.json'
JOB_FIELDS = REPO_PATH + '/assets/job_fields.json'
BLS_JOBS = REPO_PATH +'/assets/bls_gov_jobs.json'
STATE_ABBR = REPO_PATH + '/assets/state_abbr.json'

# Modified Dataset
CLEANED_JOBS = REPO_PATH + '/archive/app/clean_postings.pqt'
CATEGORIZED_JOBS = REPO_PATH + '/archive/app/categorized_postings.pqt'
TOKENIZED_JOBS = REPO_PATH +'/archive/app/tokenized_jobs.bin' 

# Models
W2V_MODEL = REPO_PATH +'/assets/models/w2v.model'
CATEGORY_VECS = REPO_PATH + '/assets/models/vectorized_categories.bin'
PREPROCESSOR = REPO_PATH + '/assets/models/preprocessor_pipe.xz'
XGB_MODEL = REPO_PATH + '/assets/models/XGBReggressor.ubj'