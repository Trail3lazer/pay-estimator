import os

REPO_PATH = os.path.abspath('')

BLS_JOBS = REPO_PATH +'/assets/bls_gov_jobs.json' #From https://www.bls.gov/ooh/a-z-index.htm
STATE_ABBR = REPO_PATH + '/assets/state_abbr.json' #This file was modified from this github gist https://gist.github.com/JeffPaine/3083347

POSTINGS = REPO_PATH + '/archive/postings.csv'
COMPANIES = REPO_PATH + '/archive/companies/companies.csv'
COMPANY_INDUSTRIES = REPO_PATH + '/archive/companies/company_industries.csv'
COMPANY_SPECIALTIES = REPO_PATH + '/archive/companies/company_specialties.csv'
BENEFITS = REPO_PATH + '/archive/jobs/benefits.csv'
JOB_SKILLS = REPO_PATH + '/archive/jobs/job_skills.csv'
JOB_INDUSTRIES = REPO_PATH + '/archive/jobs/job_industries.csv'
SALARIES = REPO_PATH + '/archive/jobs/salaries.csv'
SKILLS = REPO_PATH + '/archive/mappings/skills.csv'
INDUSTRIES = REPO_PATH + '/archive/mappings/industries.csv'

CLEANED_JOBS = REPO_PATH + '/archive/clean_postings.bin'

TOKENIZED_JOBS = REPO_PATH +'/archive/tokenized_jobs.bin'
W2V_MODEL = REPO_PATH +'/assets/w2v/w2v.model'

JOB_CATEGORIES = REPO_PATH + '/assets/job_categories.json'
CATEGORY_VECS = REPO_PATH + '/assets/w2v/vectorized_categories.bin'

CATEGORIZED_JOBS = REPO_PATH + '/archive/categorized_job_titles.bin' 

ENTEMB_MODEL = REPO_PATH + 'assets/entemb/entemb.model'