
# coding: utf-8

# Load the required libraries and data. 

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
from sklearn import model_selection, preprocessing
import xgboost as xgb
import lightgbm as lgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv("./input/train.csv", parse_dates=['timestamp'])
test = pd.read_csv('./input/test.csv', parse_dates=['timestamp'])
macro = pd.read_csv('./input/macro.csv', parse_dates=['timestamp'])
df_fixup = pd.read_excel('./input/BAD_ADDRESS_FIX.xlsx')
train_lat_lon = pd.read_csv('./input/train_lat_lon.csv')
test_lat_lon = pd.read_csv('./input/test_lat_lon.csv')
id_test = test.id
train.sample(3)
# Any results you write to the current directory are saved as output.


# In[3]:

#features_to_use_macro = ['timestamp']
#macro = macro[features_to_use_macro]

features_to_use_lat_lon = ['id', 'lat']
train_lat_lon = train_lat_lon[features_to_use_lat_lon]
test_lat_lon = test_lat_lon[features_to_use_lat_lon]
train_lat_lon.head()
train = pd.merge(train, train_lat_lon, on ='id', how = 'left')
test = pd.merge(test, test_lat_lon, on = 'id', how = 'left')


# In[4]:

y_train = train["price_doc"]
Scaler = preprocessing.MinMaxScaler()
y_train_lgb = Scaler.fit_transform(y_train)
#y_train_lgb = np.log1p(y_train)
num_train = train.shape[0]
df_all = pd.concat([train.drop(['price_doc'], axis=1), test])
#-----------------------------------
#for i in (list(df_fixup['id'])):
#    (df_all.loc[df_all['id'] == i, df_fixup.columns]) = df_fixup.loc[df_fixup['id']== i].values
#--------------------------------------------------------
df_all = pd.merge(df_all, macro, on = 'timestamp', how = 'left')
#----------------
df_all.loc[df_all['id'] == 18344, 'full_sq'] = 63
#--------------------
df_all['apartment_name']= df_all.sub_area + df_all['metro_km_avto'].astype(str)
build_year_groupby_apartment_name = df_all.groupby(['apartment_name'])['build_year'].mean()
build_year_groupby_apartment_name = pd.DataFrame({'apartment_name': build_year_groupby_apartment_name.index,
                                                  'build_year_mean': build_year_groupby_apartment_name.values})
df_all = pd.merge(df_all, build_year_groupby_apartment_name, on = 'apartment_name', how = 'left')
df_all['build_year'] = df_all['build_year'].fillna(df_all['build_year_mean'])
#------------------------------
#-------------imporve-----
df_all.loc[df_all['state'] > 4, 'state'] = 3#df_all['state'].mode()
df_all.loc[df_all['build_year'] < 3, 'build_year'] = 1691 #df_all['build_year'].mode()
df_all.loc[df_all['build_year'] == 3, 'build_year'] = 1860 #df_all['build_year'].mode()
df_all.loc[df_all['build_year'] == 71, 'build_year'] = 1860 #df_all['build_year'].mode()
df_all.loc[df_all['build_year'] == 20, 'build_year'] = 1917 #df_all['build_year'].mode()
df_all.loc[df_all['build_year'] > 4000, 'build_year'] = 2007 #df_all['build_year'].mode()
df_all.loc[df_all['full_sq'] == 5326, ['full_sq']] = 220
df_all.loc[df_all['life_sq'] == 7478, ['life_sq']] = 189
#-------------------------------------------------------------------
#-------------imporve------
df_all.loc[df_all['full_sq'] < 10, ['full_sq']] = train['full_sq'].mean()
df_all.loc[df_all['life_sq'] < 10, ['life_sq']] = train['life_sq'].mean()
df_all.loc[df_all['kitch_sq'] < 2, ['kitch_sq']] = train['kitch_sq'].mean()
df_all.loc[df_all['max_floor'] < df_all['floor'], ['max_floor']] = df_all.loc[df_all['max_floor'] < df_all['floor'], ['floor']] + 1
df_all['floor_from_top'] = df_all.apply(lambda x: x['max_floor'] - x['floor'], axis=1)
#-------------------------------------------------------------------
#--------------------new try-------------------
full_life_ratio = (df_all['full_sq'] / df_all['life_sq']).mean()
df_all['life_sq'] = df_all['life_sq'].fillna(df_all['full_sq']/ full_life_ratio)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
room_sq = df_all['life_sq'] /df_all['num_room']
df_all['num_room'] = df_all['num_room'].fillna(df_all['life_sq']/ room_sq.median())
#-----
#------------------------------------------------
kitch_ratio = df_all['full_sq'] / df_all['kitch_sq']
df_all['kitch_sq'] = df_all['kitch_sq'].fillna(df_all['full_sq']/ kitch_ratio.median())
#------
df_all['material'] = df_all['material'].fillna(df_all['material'].mode().iloc[0])
df_all['state'] = df_all['state'].fillna(df_all['state'].mode().iloc[0])
#------
df_all['max_floor'] = df_all['max_floor'].fillna(df_all['floor'] + df_all['floor_from_top'].mean())
df_all['floor_from_top'] = df_all.apply(lambda x: x['max_floor'] - x['floor'], axis=1)
#---------------
#df_all['building_age'] = df_all["timestamp"].dt.year - df_all['build_year']
df_all['building_age'] = 2020 - df_all['build_year']
df_all.loc[df_all['building_age'] < 0, 'building_age'] = np.nan
#--------2017/6/5--------new feature---------------
df_all['ratio_life_sq_full_sq'] = df_all['life_sq'] / df_all['full_sq']
df_all.loc[df_all['ratio_life_sq_full_sq'] > 1.0, 'ratio_life_sq_full_sq'] = np.nan
df_all.loc[df_all['ratio_life_sq_full_sq'] < 0, 'ratio_life_sq_full_sq'] = np.nan
df_all['ratio_floor_maxfloor'] = df_all['floor'] / df_all['max_floor']
df_all['ratio_kitch_sq_full_sq'] = df_all['kitch_sq'] / df_all['full_sq']
df_all.loc[df_all['ratio_kitch_sq_full_sq'] > 1.0, 'ratio_kitch_sq_full_sq'] = np.nan
df_all.loc[df_all['ratio_kitch_sq_full_sq'] < 0, 'ratio_kitch_sq_full_sq'] = np.nan
#---------------------------------------------------------
df_all['room_sq'] = (df_all['life_sq']-df_all['kitch_sq']) / df_all['num_room']
df_all.loc[df_all['room_sq'] < 1, 'room_sq'] = np.nan

df_all['non_living_sq'] = df_all['full_sq'] - df_all['life_sq']
df_all.loc[df_all['non_living_sq'] < 1, 'non_living_sq'] = np.nan
#-----------------------------------------------------------
df_all.drop([
    "young_male", "school_education_centers_top_20_raion", "0_17_female",
    "railroad_1line", "7_14_female", "0_17_all", "children_school","ecology", "16_29_male", 
    "mosque_count_3000", "female_f", "church_count_1000", "railroad_terminal_raion","mosque_count_5000", 
    "big_road1_1line", "mosque_count_1000", "7_14_male", "0_6_female", "oil_chemistry_raion","young_all", 
    "0_17_male", "ID_bus_terminal", "university_top_20_raion", "mosque_count_500","ID_big_road1","ID_railroad_terminal", 
    "ID_railroad_station_walk", "ID_big_road2", "ID_metro", "ID_railroad_station_avto","0_13_all", "mosque_count_2000", 
    "work_male", "16_29_all", "young_female", "work_female", "0_13_female","ekder_female", "7_14_all", "big_church_count_500",
    "leisure_count_500", "cafe_sum_1500_max_price_avg", "leisure_count_2000","office_count_500", "male_f", "nuclear_reactor_raion",
    "0_6_male", "church_count_500", "build_count_before_1920","thermal_power_plant_raion", "cafe_count_2000_na_price", 
    "cafe_count_500_price_high","market_count_2000", "museum_visitis_per_100_cap", "trc_count_500", "market_count_1000",
    "work_all", "additional_education_raion","build_count_slag", "leisure_count_1000", "0_13_male", "office_raion",
    "raion_build_count_with_builddate_info", "market_count_3000", "ekder_all", "trc_count_1000", "build_count_1946-1970",
    "office_count_1500", "cafe_count_1500_na_price", "big_church_count_5000", "big_church_count_1000", "build_count_foam",
    "church_count_1500", "church_count_3000", "leisure_count_1500","16_29_female", "build_count_after_1995", "cafe_avg_price_1500", 
    "office_sqm_1000", "cafe_avg_price_5000", "cafe_avg_price_2000","big_church_count_1500", "full_all", "cafe_sum_5000_min_price_avg",
    "office_sqm_2000", "church_count_5000","0_6_all", "detention_facility_raion", "cafe_avg_price_3000"], axis=1, inplace=True)
#---------------------------------------------------------
#df_all['full_sq_discrete'] = round(df_all['full_sq'] / 50.0 + 0.5)
#---------------------------------------------------------
df_all.drop(["id", "timestamp", "apartment_name", "build_year_mean"], axis=1, inplace = True)

#-------------------------------
for c in df_all.columns:
    if df_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_all[c].values)) 
        df_all[c] = lbl.transform(list(df_all[c].values))
#--------------------------------
print (df_all.shape)


# In[6]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
color = sns.color_palette()
f, ax = plt.subplots(figsize=(10, 15))
train_na = (df_all.isnull().sum() / len(df_all)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
sns.barplot(y=train_na.index, x=train_na,color=color[0])
plt.xlabel('% missing')


# In[6]:

x_train = df_all[0:num_train].copy().values
x_test = df_all[num_train:].copy().values
#-----Imputation transformer for completing missing values.---#
#imputer = preprocessing.Imputer()
#x_train = imputer.fit_transform(x_train)
#x_test = imputer.transform(x_test)
#---------------dont know whether effect is good--------------#


# In[7]:

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'colsample_bylevel': 0.7,
    'seed': 10
}


# In[8]:

dtrain = xgb.DMatrix(x_train, y_train_lgb)
dtest = xgb.DMatrix(x_test)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=3, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False, seed = 0)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()


# In[13]:

print (cv_output['test-rmse-mean'][len(cv_output)-1], len(cv_output))
num_boost_rounds = len(cv_output)


# In[14]:

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2_root',
    'num_leaves': 31,
    'max_depth': 5,
    'learning_rate': 0.03,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 30
}


# In[15]:

dtrain = lgb.Dataset(x_train, y_train_lgb)
cv_output2 = lgb.cv(lgb_params, dtrain, num_boost_round=1000, nfold=5, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False, seed = 0)


# In[17]:

np.exp(0.45827129375765812)+1


# In[51]:

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=1000, num_rounds=400):
    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'seed': seed_val,
        #'lambda': 5,
        #'gamma': 1.0,
        'colsample_bylevel': 0.7
    }
    num_rounds = num_rounds

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(xgb_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval = 400)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(xgb_params, xgtrain, num_rounds, verbose_eval = 50)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

def runLGB(train_X, train_y, test_X, test_y=None, seed_val = 30, num_rounds=4000):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l2_root',
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': 0,
        'seed': seed_val
    }

    lgb_train = lgb.Dataset(train_X, np.array(train_y))
    
    if test_y is not None:
        lgb_eval = lgb.Dataset(test_X, np.array(test_y))
        model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], num_boost_round=int(num_rounds), early_stopping_rounds=50, verbose_eval=1000)
    else:
        model = lgb.train(params, lgb_train, num_boost_round=int(num_rounds), verbose_eval=50)

    pred_test_y = model.predict(test_X)
    return pred_test_y, model


# 第一列

# In[52]:

xgb1_seed = 1007 #1007
xgb1_random = 1212 #1212

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=xgb1_random)
cv_train = np.zeros((x_train.shape[0]))
for train_index, test_index in kfold.split(range(x_train.shape[0])):
    preds, model = runXGB(x_train[train_index], y_train_lgb[train_index], x_train[test_index], y_train_lgb[test_index],
                          seed_val = xgb1_seed, num_rounds = 1000)
    cv_train[test_index] = preds
    
y_pred, model = runXGB(x_train, y_train_lgb, x_test, seed_val=xgb1_seed, num_rounds=300)
y_pred = y_pred
#y_pred_ = np.exp(y_pred)-1
y_pred_ = Scaler.inverse_transform(y_pred)
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_})
output.head()


# In[54]:

train_values = np.column_stack((x_train, cv_train))
test_values = np.column_stack((x_test, y_pred))


# 第二列

# In[55]:

lgb1_seed = 100 #100
lgb1_random = 1000 #1000
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=lgb1_random) #2017 0 1234
cv_train = np.zeros((x_train.shape[0]))
for train_index, test_index in kfold.split(range(x_train.shape[0])):
    preds, model = runLGB(x_train[train_index], y_train_lgb[train_index], x_train[test_index], y_train_lgb[test_index],
                          seed_val=lgb1_seed, num_rounds = 1000)
    cv_train[test_index] = preds#Scaler.inverse_transform(preds.reshape(-1,1))
    
y_pred, model = runLGB(x_train, y_train_lgb, x_test, seed_val=lgb1_seed, num_rounds=400)
y_pred_ = Scaler.inverse_transform(y_pred) #98
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_})
output.head()


# In[57]:

train_values = np.column_stack((train_values, cv_train))
test_values = np.column_stack((test_values, y_pred))


# 第三列

# In[58]:

lgb2_seed = 711 #711
lgb2_random = 2017 #2017
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state= lgb2_random) #678
cv_train = np.zeros((x_train.shape[0]))
for train_index, test_index in kfold.split(range(x_train.shape[0])):
    preds, model = runLGB(x_train[train_index], y_train_lgb[train_index], x_train[test_index], y_train_lgb[test_index],
                          seed_val=lgb2_seed, num_rounds = 1000)
    cv_train[test_index] = preds
    
y_pred, model = runLGB(x_train, y_train_lgb, x_test, seed_val=lgb2_seed, num_rounds=600)
y_pred_ = Scaler.inverse_transform(y_pred)
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_})
output.head()


# In[60]:

train_values = np.column_stack((train_values, cv_train))
test_values = np.column_stack((test_values, y_pred))


# 第四列

# In[63]:

lgb3_seed = 1000 #1000
lgb3_random = 2124 # 0
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state= lgb3_random) #678
cv_train = np.zeros((x_train.shape[0]))
for train_index, test_index in kfold.split(range(x_train.shape[0])):
    preds, model = runLGB(x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index],
                          seed_val=lgb3_seed, num_rounds = 1000)
    cv_train[test_index] = preds

y_pred, model = runLGB(x_train, y_train, x_test, seed_val=lgb3_seed, num_rounds=400)
y_pred_ = np.exp(y_pred)-1
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_})
output.head()


# In[91]:

y_pred_ = np.exp(y_pred*1.002)-1
print (y_pred_.mean())
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_})
output.head()


# In[92]:

train_values = np.column_stack((train_values, cv_train))
test_values = np.column_stack((test_values, y_pred*1.002))


# 第五列

# In[93]:

xgb2_seed = 0 #0
xgb2_random = 2017 #2017

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=xgb2_random)
cv_train = np.zeros((x_train.shape[0]))
for train_index, test_index in kfold.split(range(x_train.shape[0])):
    preds, model = runXGB(x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index],
                          seed_val = xgb2_seed, num_rounds = 1000)
    cv_train[test_index] = preds
    
y_pred, model = runXGB(x_train, y_train, x_test, seed_val=xgb2_seed, num_rounds=400)
y_pred = y_pred# * .969
y_pred_ = np.exp(y_pred)-1
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_})
output.head()


# In[100]:

y_pred_ = np.exp(y_pred*1.0019)-1
print (y_pred_.mean())
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_})
output.head()


# In[101]:

train_values = np.column_stack((train_values, cv_train))
test_values = np.column_stack((test_values, y_pred))


# 第六列

# In[102]:

xgb3_seed = 2237 #2335
xgb3_random = 2131 #1000

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=xgb3_random)
cv_train = np.zeros((x_train.shape[0]))
for train_index, test_index in kfold.split(range(x_train.shape[0])):
    preds, model = runXGB(x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index],
                          seed_val = xgb3_seed, num_rounds = 1000)
    cv_train[test_index] = preds
    
y_pred, model = runXGB(x_train, y_train, x_test, seed_val=xgb3_seed, num_rounds=300)
y_pred = y_pred
y_pred_ = np.exp(y_pred)-1
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_})
output.head()


# In[106]:

y_pred_ = np.exp(y_pred*1.0022)-1
print (y_pred_.mean())
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_})
output.head()


# In[107]:

train_values = np.column_stack((train_values, cv_train))
test_values = np.column_stack((test_values, y_pred))


# 第二层训练

# XGB

# In[122]:

import random
size = 20
sum_ = np.zeros(7662)
for i in range(size):
    xgb_seed = random.randint(0, 87562) #300
    #lgb_seed = random.randint(12323, 232373)
    y_pred_res, model_res = runXGB(train_values, y_train_lgb, test_values, seed_val=xgb_seed, num_rounds= 175)#175 200
    #y_pred_res = np.exp(y_pred_res)-1
    #y_pred_res2, model_res = runLGB(train_values, y_train_lgb, test_values,seed_val=lgb_seed, num_rounds= 300)
    #y_pred_res2 = np.exp(y_pred_res2)-1
    sum_ += y_pred_res / size
    #sum_ += ((0.75*y_pred_res+0.25*y_pred_res2) / size)
y_pred_res = sum_
y_pred_res = Scaler.inverse_transform(y_pred_res)
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred_res})
output


# In[138]:

output.to_csv('single_sub.csv', index = False)
#output.to_csv('yearMYxgb2.csv', index = False)

