#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import pandas as pd
import sklearn


# In[3]:


sklearn.__version__


# In[5]:


year = 2023
month = 3

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


# In[20]:


get_ipython().system('mkdir output')


# In[7]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[8]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# In[10]:


df = read_data(input_file)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[11]:


df.head()


# In[12]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[13]:


y_pred


# ### Q1. Standard deviation

# In[17]:


y_pred.std()


# ### Q2. Preparing the output

# In[18]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred


# In[21]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[23]:


get_ipython().system('ls -lh output')


# In[ ]:




