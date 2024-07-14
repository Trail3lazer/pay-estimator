# Job Posting Pay Estimator

In this notebook we use the [LinkedIn Job Postings (2023 - 2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) kaggle dataset to estimate job pay based on a job title and the state it is in. As we analyze the data, we use the [Plotly](https://plotly.com/python/) library for visualizations. We train the XGBRegressor model from the [DMLC XGBoost](https://xgboost.readthedocs.io/) library to estimate the job pay. The XGBoost model only allows numeric inputs. So, we use the [Sci-kit Learn](https://scikit-learn.org/) library to pipe our data through transformers that translate the job titles and states into numbers and interact with XGBoost. We use the Sci-kit Learn OneHotEncoder to transform the state. Since the job titles have such a high cardinality, we need to represent them differently. So, we use the [Gensim](https://radimrehurek.com/gensim/) library Word2Vec model to create word embedding vectors representing the job titles for XGBoost. After training XGBRegressor, we use IPython to make a small user interface to estimate the job pay based on the job title and state.

#### Notebook URL: https://colab.research.google.com/github/Trail3lazer/pay-estimator/blob/main/pay-estimator.ipynb

## Sources

| name | url | modified |
| ---- | --- | -------- |
| LinkedIn Job Postings (2023 - 2024) Dataset | https://www.kaggle.com/datasets/arshkon/linkedin-job-postings | Y | 
| DMLC XGBoost | https://xgboost.readthedocs.io/ | N |
| Sci-kit Learn (sklearn) | https://scikit-learn.org/ | N |
| Gensim | https://radimrehurek.com/gensim/ | N |
| Plotly | https://plotly.com/python/ | N |
| Other python dependencies | ... | N |
| job_fields.json | https://www.bls.gov/oes/current/oes_stru.htm | Y |
| bls_gob_jobs.json and job_categories.json | https://www.bls.gov/ooh/a-z-index.htm | Y |
| state_abbr.json | https://gist.github.com/JeffPaine/3083347 | Y |
| State matching regex | https://sigpwned.com/2023/06/29/regex-for-50-us-states/ | Y |
| Full-time and part-time hours | https://www.bls.gov/charts/american-time-use/emp-by-ftpt-job-edu-h.htm | Y |
| Number of vacation days | [https://www.bls.gov/ebs/factsheets/paid-vacations](https://www.bls.gov/ebs/factsheets/paid-vacations.htm#:~:text=The%20number%20of%20vacation%20days,19%20days%20of%20paid%20vacation) | Y |
| Number of federal holidays | https://www.ca2.uscourts.gov/clerk/calendars/federal_holidays.html | Y |
