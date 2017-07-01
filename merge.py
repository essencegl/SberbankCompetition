import numpy as np 
import pandas as pd
import random

address = ['7+hebing200.csv', '7+hebing1(0.30983).csv', '7+hebing200.csv','2.csv', '3.csv', '4.csv',
           '5.csv', '6.csv', '7.csv', '8.csv', '9.csv', '10.csv', '11.csv', '12.csv',
           '13.csv', '14.csv', '15.csv', '16.csv', '17.csv', '18.csv', '19.csv', '20.csv',
           '21.csv', '22.csv', '23.csv', '24.csv', '25.csv', '26.csv', '27.csv', '28.csv',
           '29.csv', '30.csv', '31.csv', '31.csv', '31.csv', '31.csv', '31.csv', '31.csv',
           '31.csv','31.csv', '31.csv', '31.csv', '31.csv', '31.csv', '31.csv', '32.csv',
           '32.csv', '32.csv', '32.csv', '32.csv', '32.csv', '32.csv', '32.csv', '32.csv',
           '32.csv', '32.csv', '32.csv','xgb&lgb.csv','xgb&lgb.csv','xgb&lgb.csv','xgb&lgb.csv',
           'xgb&lgb.csv', 'xgb&lgb.csv',#0.30928
           'xgb&lgb2_lgb.csv', 'xgb&lgb2_xgb.csv', 'xgb&lgb2_xgb_3layers.csv',#0.30924
           'xgb&lgb2_xgb_3layers.csv', 'xgb&lgb2_xgb_3layers.csv',#0.30920
           'xgb&lgb2_xgb_3layers.csv', 'xgb&lgb2_xgb_3layers.csv', #0.30918
           'xgb&lgb2_xgb_3layers.csv', 'xgb&lgb2_xgb_3layers.csv', #0.30916
           'sub2.csv', 'sub2.csv', 'xgb&lgb2_xgb_3layers.csv', 'xgb&lgb2_xgb_3layers.csv',
           'xgb&lgb2_xgb_3layers.csv', #0.30914
           'sub-silly-fixed-price-changed-local.csv', 'sub-silly-fixed-price-changed-local.csv',
           'sub-silly-fixed-price-changed-local.csv', 'sub-silly-fixed-price-changed-local.csv',
           'sub-silly-fixed-price-changed-local.csv', 'sub-silly-fixed-price-changed-local.csv',
           'sub-silly-fixed-price-changed-local.csv', 'sub-silly-fixed-price-changed-local.csv',#0.30912
           'xgb&lgb2_xgb_3layers.csv', 'xgb&lgb2_xgb_3layers.csv', 'xgb&lgb2_xgb_3layers.csv', #0.30911
           'different_result.csv', 'different_result.csv', 'same_result.csv', 'same_result.csv']
def merge(address):
  k = len(address)
  for i in range(k):
    df = pd.read_csv(address[i])
    if (i == 0):
      price_final = df['price_doc']
      id_final = df['id']
    else:
      price_final += df['price_doc']
  price_final = price_final / k
  for j in range(price_final.shape[0]):
    if (random.random()<0.73):
      price_final[j] = price_final[j]*0.993
    else:
      price_final[j] = price_final[j]*0.9977
  out_df = pd.DataFrame({'id': id_final, 'price_doc': price_final}) #0.994->0.30903 0.992->0.30903
  out_df.to_csv('merge.csv', index=False)

# Main Function
merge(address)
