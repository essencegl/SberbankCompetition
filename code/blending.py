
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
get_ipython().magic('matplotlib inline')
df1 = pd.read_csv('0.30898.csv')
df2 = pd.read_csv('same_result_new.csv')
f, ax = plt.subplots(figsize=(500, 8))
ts_df = df1['price_doc']
plt.plot(df1['id'], df1, color='r')
ax.set(title='id')

plt.plot(df2['id'], df2, color='b')
ax.set(title='id')


# In[4]:

f, ax = plt.subplots(figsize=(100, 8))
df_sub = pd.DataFrame()
df_sub['id'] = df1['id']
df_sub['price_doc'] = df1['price_doc'] - df2['price_doc']
plt.plot(df_sub['id'], df_sub, color='b')
ax.set(title='id')


# Verision - 0.30844

# In[11]:

#0.30844
id_r = df1['id']
out_df = pd.DataFrame()
out_df['id'] = id_r
value = np.zeros(id_r.shape[0])
count = np.zeros(4)
for i in range(id_r.shape[0]):
    if (abs(df_sub['price_doc'][i]) <= 200000):
        value[i] = (0.52*df1['price_doc'][i] + 0.48*df2['price_doc'][i])
        count[0] += 1
    elif (abs(df_sub['price_doc'][i]) <= 700000):
        value[i] = (0.60*df1['price_doc'][i] + 0.40*df2['price_doc'][i])
        count[1] += 1
    elif (abs(df_sub['price_doc'][i]) <= 1000000):
        value[i] = (0.66*df1['price_doc'][i] + 0.34*df2['price_doc'][i])
        count[2] += 1
    else:
        value[i] = df1['price_doc'][i]
        count[3] += 1
print (count)
out_df['price_doc'] = value
out_df.to_csv('0.30844.csv', index=False)


# Version - 0.30841

# In[12]:

id_r = df1['id']
out_df = pd.DataFrame()
out_df['id'] = id_r
value = np.zeros(id_r.shape[0])
count = np.zeros(10)
for i in range(id_r.shape[0]):
    if (abs(df_sub['price_doc'][i]) <= 100000):
        value[i] = (0.52*df1['price_doc'][i] + 0.48*df2['price_doc'][i])
        count[0] += 1        
    if (abs(df_sub['price_doc'][i]) <= 400000):
        value[i] = (0.6*df1['price_doc'][i] + 0.4*df2['price_doc'][i])
        count[1] += 1
    elif (abs(df_sub['price_doc'][i]) <= 700000):
        value[i] = (0.68*df1['price_doc'][i] + 0.32*df2['price_doc'][i])
        count[2] += 1
    elif (abs(df_sub['price_doc'][i]) <= 1000000):
        value[i] = (0.72*df1['price_doc'][i] + 0.28*df2['price_doc'][i])
        count[3] += 1
    else:
        value[i] = (0.76*df1['price_doc'][i] + 0.24*df2['price_doc'][i])
        count[4] += 1
out_df['price_doc'] = value
out_df.to_csv('0.30841.csv', index=False)


# Version - 0.30837

# In[13]:

id_r = df1['id']
out_df = pd.DataFrame()
out_df['id'] = id_r
value = np.zeros(id_r.shape[0])
count = np.zeros(10)
for i in range(id_r.shape[0]):
    if (abs(df_sub['price_doc'][i]) <= 100000):
        value[i] = (0.52*df1['price_doc'][i] + 0.48*df2['price_doc'][i])
        count[0] += 1        
    elif (abs(df_sub['price_doc'][i]) <= 400000):
        value[i] = (0.6*df1['price_doc'][i] + 0.4*df2['price_doc'][i])
        count[1] += 1
    elif (abs(df_sub['price_doc'][i]) <= 700000):
        value[i] = (0.68*df1['price_doc'][i] + 0.32*df2['price_doc'][i])
        count[2] += 1
    elif (abs(df_sub['price_doc'][i]) <= 1000000):
        value[i] = (0.76*df1['price_doc'][i] + 0.24*df2['price_doc'][i])
        count[3] += 1
    elif (abs(df_sub['price_doc'][i]) <= 1300000):
        value[i] = (0.84*df1['price_doc'][i] + 0.16*df2['price_doc'][i])
        count[4] += 1
    elif (abs(df_sub['price_doc'][i]) <= 1600000):
        value[i] = (0.92*df1['price_doc'][i] + 0.08*df2['price_doc'][i])
        count[5] += 1
    elif (abs(df_sub['price_doc'][i]) <= 1900000):
        value[i] = (0.98*df1['price_doc'][i] + 0.02*df2['price_doc'][i])
        count[6] += 1
    else:
        value[i] = df1['price_doc'][i]# + df2['price_doc'][i])
        count[7] += 1
out_df['price_doc'] = value
out_df.to_csv('0.30837.csv', index=False)
print (out_df.price_doc.mean())
print (count)
out_df


# In[14]:

out_df.to_csv('finalsub.csv', index=False)

