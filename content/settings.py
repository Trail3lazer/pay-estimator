import os

REPO_PATH = os.path.abspath('.')

ARCHIVE_PATH = REPO_PATH + '/archive'
APP_ARCHIVE_PATH = ARCHIVE_PATH + '/app'
ASSETS_PATH = REPO_PATH + '/assets'
MODELS_PATH = ASSETS_PATH + '/models'
ARCHIVE_EXT = '.csv'

# Dataset
POSTINGS = ARCHIVE_PATH + '/postings' + ARCHIVE_EXT
COMPANIES = ARCHIVE_PATH + '/companies/companies' + ARCHIVE_EXT
COMPANY_INDUSTRIES = ARCHIVE_PATH + '/companies/company_industries' + ARCHIVE_EXT
COMPANY_SPECIALITIES = ARCHIVE_PATH + '/companies/company_specialities' + ARCHIVE_EXT
EMPLOYEES = ARCHIVE_PATH + '/companies/employee_counts' + ARCHIVE_EXT
BENEFITS = ARCHIVE_PATH + '/jobs/benefits' + ARCHIVE_EXT
JOB_SKILLS = ARCHIVE_PATH + '/jobs/job_skills' + ARCHIVE_EXT
JOB_INDUSTRIES = ARCHIVE_PATH + '/jobs/job_industries' + ARCHIVE_EXT
SALARIES = ARCHIVE_PATH + '/jobs/salaries' + ARCHIVE_EXT
SKILLS = ARCHIVE_PATH + '/mappings/skills' + ARCHIVE_EXT
INDUSTRIES = ARCHIVE_PATH + '/mappings/industries' + ARCHIVE_EXT

# Additional data
JOB_CATEGORIES = ASSETS_PATH + '/job_categories.json'
JOB_FIELDS = ASSETS_PATH + '/job_fields.json'
BLS_JOBS = ASSETS_PATH + '/bls_gov_jobs.json'
STATE_ABBR = ASSETS_PATH + '/state_abbr.json'

# Modified Dataset
CLEANED_JOBS = APP_ARCHIVE_PATH + '/clean_postings.pqt'
CATEGORIZED_JOBS = APP_ARCHIVE_PATH + '/categorized_postings.pqt'
TOKENIZED_JOBS = APP_ARCHIVE_PATH + '/tokenized_jobs.bin' 

# Models
W2V_MODEL = MODELS_PATH + '/w2v.bin'
CATEGORY_VECS = MODELS_PATH + '/vectorized_categories.bin'
XGB_MODEL = MODELS_PATH + '/XGBReggressor.ubj'

def set_archive_path(path: str):
    APP_ARCHIVE_PATH = path