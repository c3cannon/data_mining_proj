# -*- coding: utf-8 -*-
"""Code- Baseline models

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U8Y094uLGGs4rKHDBOAFn5NrXo416eIi
"""

!pip install catboost
!pip install ipywidgets
!jupyter nbextension enable --py widgetsnbextension

# Commented out IPython magic to ensure Python compatibility.
# PLACE ALL IMPORTS HERE
import pandas as pd
import sys
from google.colab import drive
import sklearn
import sklearn.impute, sklearn.pipeline
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import uniform, randint


from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

from IPython.display import display
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

"""# Initialization"""

# MOUNT YOUR GOOGLE DRIVE 
# (It only mounts to your private Virtual Machine, it doesn't expose your drive to anyone else)
drive.mount("/content/drive", force_remount=True)

# LOAD THE FILE
df = pd.read_csv("/content/drive/My Drive/Data Mining Project - Share/Data/combined.csv")
# Converting the column to DateTime format
df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
df = df.set_index('Date')
df.head()

# Convert to numerical values
for col in df.columns:
  if (col != "freq"):
    df[col] = pd.to_numeric(df[col], errors="coerce")

display(df.head())

"""# Exploratory Data Analysis"""

display(df)

print("First valid Spend ACF:",df["spend_acf"].first_valid_index())
print("First valid Test:", df["test_count"].first_valid_index())
print("First valid bg_posts:", df["bg_posts"].first_valid_index())

"""# Preprocessing

## Filtering
"""

# DO PREPROCESSING

# Drop January and November
df_simple = df[df["month"].gt(1) & df["month"].lt(11)].copy()
df_simple.head()

"""## Imputing/Filling

Stategy: Don't use Simple Imputer, which takes the mean/median/mode/etc. across all records in a feature. This would lose the time sensitive data. Here we are using "method = time" for imputation. Feel free to change it to linear if you want or anything else and test it
"""

# First fill COVID data
covid_ids = ["death", "case", "test", "positives"]
# If March or earlier, fill NAs with zero
for col in df_simple.columns:
  for covid_id in covid_ids:
    if covid_id in col:
      df_simple.loc[df_simple["month"]<=3, col] = df_simple.loc[df_simple["month"]<=3, col].fillna(0, inplace=False)
      # Also force all to be >= 0
      df_simple.loc[:, col] = df_simple.loc[:, col].clip(lower=0, inplace=False)
      #df_simple.loc[col, df_simple[col].isna()] = 0


#df_simple = df_simple.drop(['year','month','day','freq'],axis=1)
## Impute the DATA
count = df_simple.isna().sum()
print(count)
df_simple = df_simple.interpolate(method = 'time')

#df_slinear = df_simple.assign(InterpolateSLinear=df_simple.target.interpolate(method='slinear'))
count_2 = df_simple.isna().sum()
print(count_2)

## This is just a function used to get the covid delays, 
## just pass to it a delay vector and it can delay the covid data by any element in the vector
def df_derived_by_shift(df,lag=0):
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in lag:
        for x in list(df.columns):
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)    
        i = 0
        for c in columns:
            dfn[c] = df[k].shift(periods=lag[i]*51) ## this is because we want to shift all the 51 states
            i = i+1
        df = pd.concat([df, dfn], axis=1)
    return df

# Drop cumulative COVID data (better to use rates/new counts)
covid_columns = ["case_rate", 	
                 "death_rate", 	
                 "test_rate", 	
                 "new_positives_rate",
                 "new_case_rate", 	
                 "new_death_rate", 	
                 "new_test_rate", 	
                 "new_case_count", 	
                 "new_death_count", 	
                 "new_test_count"]

states = ["statefips"]  

spending_columns = ["spend_acf",	"spend_aer",	"spend_all",	"spend_apg",	"spend_grf",	"spend_hcs","spend_tws",
                    "spend_all_inchigh",	"spend_all_inclow",	"spend_all_incmiddle"]

revenues_columns = ["revenue_all",	"revenue_ss40",	"revenue_ss60",	"revenue_ss65",
                    "revenue_ss70",	"merchants_all",	"merchants_ss40",	"merchants_ss60",	"merchants_ss65",	"merchants_ss70"]

employment_columns = ["emp_combined",	"emp_combined_inclow",	"emp_combined_incmiddle", "emp_combined_inchigh",	"emp_combined_ss40",	"emp_combined_ss60",
                      "emp_combined_ss65",	"emp_combined_ss70","initclaims_count_regular",	"initclaims_rate_regular",	"contclaims_count_regular",	
                      "contclaims_rate_regular",	"initclaims_count_pua",	"contclaims_count_pua",	"contclaims_count_peuc",	"initclaims_rate_pua",
                      "contclaims_rate_pua",	"contclaims_rate_peuc","initclaims_count_combined",	"contclaims_count_combined",	"initclaims_rate_combined",
                      "contclaims_rate_combined",	"bg_posts","bg_posts_ss30",	"bg_posts_ss55",	"bg_posts_ss60",	"bg_posts_ss65",	"bg_posts_ss70",
                      "bg_posts_jz1",	"bg_posts_jzgrp12","bg_posts_jz2","bg_posts_jz3",	"bg_posts_jzgrp345",	"bg_posts_jz4",	"bg_posts_jz5"]


df_covid = df_simple.filter(covid_columns, axis=1)
df_states = df_simple.filter(states, axis=1)
df_spending = df_simple.filter(spending_columns, axis=1)
df_revenues = df_simple.filter(revenues_columns, axis=1)
df_employment = df_simple.filter(employment_columns, axis=1)

econ_frames = [df_spending, df_revenues,df_employment]
df_econ = pd.concat(econ_frames,axis=1)

display(df_econ.head())
display(df_covid.head())

## This is where we use the previous function to delay the covid data/ Econ data which are already imputed
## If you also want to use the econ delayed data as an input, comment out the commented commands
delay_covid = [7,14,21]
delay_econ = [7,14,21]

df_covid = df_derived_by_shift(df_covid, delay_covid)
df_econ = df_derived_by_shift(df_econ, delay_econ)
display(df_covid.head())

df_covid.to_csv('data.csv')
!cp data.csv "drive/My Drive/"

## As you can see now we have NANs that arises from 2 things:
## 1- The imputation method won't fill all of them
## 2- Shifting the data will introduce NAN entries
## Let's remove all of the NAN entries now. I am appending the data again because we need all the data to have the same index
frames = [df_states, df_covid, df_econ]
df_new = pd.concat(frames,axis=1)
df_new = df_new.dropna()
display(df_new.head())

frames2 = [df_states, df_covid, df_econ]
df_covid_w_state = pd.concat(frames2,axis=1)
df_covid_w_state = df_covid_w_state.dropna()
display(df_covid_w_state.head())

non_covid2 = states + spending_columns +  revenues_columns + employment_columns

## Save all the preprocessed data in drive
non_covid = states + spending_columns +  revenues_columns + employment_columns

df_states = df_new.filter(states, axis=1)
df_spending = df_new.filter(spending_columns, axis=1)
df_revenues = df_new.filter(revenues_columns, axis=1)
df_employment = df_new.filter(employment_columns, axis=1)

df_covid = df_new.drop(non_covid, axis=1)


econ_frames = [df_spending, df_revenues,df_employment]
df_econ = pd.concat(econ_frames,axis=1)

display(df_covid.head())
display(df_econ.head())

df_states.to_csv('df_states.csv')
!cp df_states.csv "/content/drive/My Drive/Data Mining Project - Share/Data"

df_covid.to_csv('df_covid.csv')
!cp df_covid.csv "/content/drive/My Drive/Data Mining Project - Share/Data"

df_spending.to_csv('df_spending.csv')
!cp df_spending.csv "/content/drive/My Drive/Data Mining Project - Share/Data"

df_revenues.to_csv('df_revenues.csv')
!cp df_revenues.csv "/content/drive/My Drive/Data Mining Project - Share/Data"

df_employment.to_csv('df_employment.csv')
!cp df_employment.csv "/content/drive/My Drive/Data Mining Project - Share/Data"

df_econ.to_csv('df_econ.csv')
!cp df_econ.csv "/content/drive/My Drive/Data Mining Project - Share/Data"

def regression_results(y_true, y_pred):
# Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    #mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    #print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

new_frames_covid = [df_states,df_covid]
new_frames_econ = [df_states,df_econ]

df_covid_new = pd.concat(new_frames_covid,axis=1)
df_econ_new = pd.concat(new_frames_econ,axis=1)

display(df_covid_new.head())
display(df_econ_new.head())

df_covid.to_csv('df_covid_econ_w_state.csv')
!cp df_covid_econ_w_state.csv "/content/drive/My Drive/Data Mining Project - Share/Data"

df_econ.to_csv('df_econ_w_state.csv')
!cp df_econ_w_state.csv "/content/drive/My Drive/Data Mining Project - Share/Data"

models = [RandomForestRegressor(),KNeighborsRegressor(), xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
          LGBMRegressor(), CatBoostRegressor(verbose=False)]

model_names = ['RF','KNR','XGB','LGBM','CatBoost']

param_RF = {'n_estimators': [20],'max_depth' : [5,10,15] }
param_KNR = { 'n_neighbors':[5,10,15,20,25] }
param_XGB = {}
param_Cat = {}
param_lgbm = {}
params = [param_RF, param_KNR, param_XGB, param_lgbm, param_Cat]


for i in range(len(models)):
  orig_stdout = sys.stdout
  # path = '/content/drive/My Drive/Data Mining Project - Share/Data/'+ model_names[i] + '-Results-Covid-Econ.txt'
  path = '/content/drive/My Drive/Data Mining Project - Share/Data/'+ model_names[i] + '-Results-Covid.txt'
  f = open(path, 'w')
  sys.stdout = f
  for column in df_econ_new.columns:
      if column != 'statefips':
        means = df_econ_new.loc[:'2020-07'].groupby('statefips')[column].mean()
        df_econ_encoded = df_econ_new.copy()
        temp = df_econ_encoded['statefips'].map(means)
        df_covid_encoded = df_covid_new.copy()
        df_covid_encoded['statefips'] = temp

        ## Preprocess the data using StandardScaler
        X_train = df_covid_encoded[:'2020-07']
        X_test = df_covid_encoded['2020-08-01':'2020-08-31']
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        y_train = df_econ_encoded.loc[:'2020-07', column]
        y_test = df_econ_encoded.loc['2020-8-01':'2020-08-31':, column]

        tscv = TimeSeriesSplit(n_splits=5)
        gsearch = GridSearchCV(estimator=models[i], cv=tscv, param_grid=params[i], scoring = 'r2')


        gsearch.fit(X_train, y_train)
        best_score = gsearch.best_score_
        best_model = gsearch.best_estimator_

        y_pred = best_model.predict(X_test)

        # print('Regression results Using '+ model_names[i]+ ' Regressor on', column , 'using both covid and econ data are:')
        print('Regression results Using '+ model_names[i]+ ' Regressor on', column , 'using covid data are:')
        regression_results(y_test, y_pred)
  sys.stdout = orig_stdout
  f.close()

## Note that CatBoost was failing when running the code on the whole econ+covid data
## This is why we run it only on the important features

columns = ['spend_all','revenue_all','merchants_all','emp_combined']

orig_stdout = sys.stdout
path = '/content/drive/My Drive/Data Mining Project - Share/Data/CatBoost-Results-Covid-Econ-new.txt'
f = open(path, 'w')
sys.stdout = f
for column in columns:
        if column != 'statefips':
          means = df_econ_new.loc[:'2020-07'].groupby('statefips')[column].mean()
          df_econ_encoded = df_econ_new.copy()
          temp = df_econ_encoded['statefips'].map(means)
          df_covid_encoded = df_covid_new.copy()
          df_covid_encoded['statefips'] = temp

          ## Preprocess the data using StandardScaler
          X_train = df_covid_encoded[:'2020-07']
          X_test = df_covid_encoded['2020-08-01':'2020-08-31']
          X_scaler = StandardScaler()
          X_train = X_scaler.fit_transform(X_train)
          X_test = X_scaler.transform(X_test)

          y_train = df_econ_encoded.loc[:'2020-07', column]
          y_test = df_econ_encoded.loc['2020-8-01':'2020-08-31':, column]
          
          
          tscv = TimeSeriesSplit(n_splits=5)
          gsearch = GridSearchCV(estimator=CatBoostRegressor(verbose = False), cv=tscv, param_grid={}, scoring = 'r2')


          gsearch.fit(X_train, y_train)
          best_score = gsearch.best_score_
          best_model = gsearch.best_estimator_

          y_pred = best_model.predict(X_test)

          print('Regression results Using CatBoost Regressor on', column , 'using both covid and econ data are:')
          regression_results(y_test, y_pred)
sys.stdout = orig_stdout
f.close()

models = [RandomForestRegressor(), xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
          LGBMRegressor(), CatBoostRegressor(verbose=False)]

model_names = ['RF','XGB','LGBM','CatBoost']

param_RF = {'n_estimators': [20],'max_depth' : [5,10,15] }

param_XGB = {}
param_Cat = {}
param_lgbm = {}
params = [param_RF, param_XGB, param_lgbm, param_Cat]

columns = ['spend_all','revenue_all','merchants_all','emp_combined']

for i in range(len(models)):

  for column in columns:
      if column != 'statefips':
        means = df_econ_new.loc[:'2020-07'].groupby('statefips')[column].mean()
        df_econ_encoded = df_econ_new.copy()
        temp = df_econ_encoded['statefips'].map(means)
        df_covid_encoded = df_covid_new.copy()
        df_covid_encoded['statefips'] = temp

        ## Preprocess the data using StandardScaler
        X_train = df_covid_encoded[:'2020-07']
        X_test = df_covid_encoded['2020-08-01':'2020-08-31']
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        y_train = df_econ_encoded.loc[:'2020-07', column]
        y_test = df_econ_encoded.loc['2020-8-01':'2020-08-31':, column]

        tscv = TimeSeriesSplit(n_splits=5)
        gsearch = GridSearchCV(estimator=models[i], cv=tscv, param_grid=params[i], scoring = 'r2')


        gsearch.fit(X_train, y_train)
        best_score = gsearch.best_score_
        best_model = gsearch.best_estimator_
        
        features = list(df_covid_encoded.columns.values)

        importances_full = best_model.feature_importances_

        indices_full = np.argsort(importances_full)

        indices_full = indices_full[len(indices_full)-5:len(indices_full)]

        title = 'Top 5 features when using '+ model_names[i]+ ' on COVID data to predict '+ str(column)
        plt.title(title)
        plt.barh(range(len(indices_full)), importances_full[indices_full], color='b', align='center')
        plt.yticks(range(len(indices_full)), [features[i] for i in indices_full])
        plt.xlabel('Relative Importance')
        plt.show()

models = [RandomForestRegressor(), xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
          LGBMRegressor(), CatBoostRegressor(verbose=False)]

model_names = ['RF','XGB','LGBM','CatBoost']

param_RF = {'n_estimators': [20],'max_depth' : [5,10,15] }

param_XGB = {}
param_Cat = {}
param_lgbm = {}
params = [param_RF, param_XGB, param_lgbm, param_Cat]

columns = ['spend_all','revenue_all','merchants_all','emp_combined']

for i in range(len(models)):

  for column in columns:
      if column != 'statefips':
        means = df_econ_new.loc[:'2020-07'].groupby('statefips')[column].mean()
        df_econ_encoded = df_econ_new.copy()
        temp = df_econ_encoded['statefips'].map(means)
        df_covid_encoded = df_covid_new.copy()
        df_covid_encoded['statefips'] = temp

        ## Preprocess the data using StandardScaler
        X_train = df_covid_encoded[:'2020-07']
        X_test = df_covid_encoded['2020-08-01':'2020-08-31']
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        y_train = df_econ_encoded.loc[:'2020-07', column]
        y_test = df_econ_encoded.loc['2020-8-01':'2020-08-31':, column]

        tscv = TimeSeriesSplit(n_splits=5)
        gsearch = GridSearchCV(estimator=models[i], cv=tscv, param_grid=params[i], scoring = 'r2')


        gsearch.fit(X_train, y_train)
        best_score = gsearch.best_score_
        best_model = gsearch.best_estimator_
        
        features = list(df_covid_encoded.columns.values)

        importances_full = best_model.feature_importances_

        indices_full = np.argsort(importances_full)

        indices_full = indices_full[len(indices_full)-5:len(indices_full)]

        title = 'Top 5 features when using '+ model_names[i]+ ' on COVID & Econ data to predict '+ str(column)
        plt.title(title)
        plt.barh(range(len(indices_full)), importances_full[indices_full], color='b', align='center')
        plt.yticks(range(len(indices_full)), [features[i] for i in indices_full])
        plt.xlabel('Relative Importance')
        plt.show()