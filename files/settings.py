import os

REPO_PATH = os.path.abspath('.')

ARCHIVE_PATH = REPO_PATH + '/.archive'
APP_ARCHIVE_PATH = ARCHIVE_PATH + '/app'
ASSETS_PATH = REPO_PATH + '/assets'
MODELS_PATH = ASSETS_PATH + '/models'

# Dataset
POSTINGS = ARCHIVE_PATH + '/postings.csv'
COMPANIES = ARCHIVE_PATH + '/companies/companies.csv'
COMPANY_INDUSTRIES = ARCHIVE_PATH + '/companies/company_industries.csv'
COMPANY_SPECIALITIES = ARCHIVE_PATH + '/companies/company_specialities.csv'
EMPLOYEES = ARCHIVE_PATH + '/companies/employee_counts.csv'
BENEFITS = ARCHIVE_PATH + '/jobs/benefits.csv'
JOB_SKILLS = ARCHIVE_PATH + '/jobs/job_skills.csv'
JOB_INDUSTRIES = ARCHIVE_PATH + '/jobs/job_industries.csv'
SALARIES = ARCHIVE_PATH + '/jobs/salaries.csv'
SKILLS = ARCHIVE_PATH + '/mappings/skills.csv'
INDUSTRIES = ARCHIVE_PATH + '/mappings/industries.csv'

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
W2V_MODEL = MODELS_PATH + '/w2v.model'
CATEGORY_VECS = MODELS_PATH + '/vectorized_categories.bin'
PREPROCESSOR = MODELS_PATH + '/preprocessor_pipe.xz'
XGB_MODEL = MODELS_PATH + '/XGBReggressor.ubj'

def set_archive_path(path: str):
    APP_ARCHIVE_PATH = path