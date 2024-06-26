import os

REPO_PATH = os.path.abspath('')

BLS_JOBS = REPO_PATH +'/assets/bls_gov_jobs.json' #From https://www.bls.gov/ooh/a-z-index.htm
STATE_ABBR = REPO_PATH + '/assets/state_abbr.json' #This file was modified from this github gist https://gist.github.com/JeffPaine/3083347

ORIGINAL_CSV = REPO_PATH + '/archive/postings.csv'
CLEANED_JOBS = REPO_PATH + '/archive/clean_postings.bin'

TOKENIZED_JOBS = REPO_PATH +'/archive/tokenized_jobs.bin'
W2V_MODEL = REPO_PATH +'/assets/w2v/w2v.model'

CATEGORY_VECS = REPO_PATH + '/assets/w2v/vectorized_categories.bin'

CATEGORIZED_JOBS = REPO_PATH + '/archive/categorized_job_titles.bin' 