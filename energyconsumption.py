import os
import gc
import copy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold ,RepeatedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.style.use('ggplot')
import seaborn as sns
from scipy import stats
import shap
from sklearn.preprocessing import StandardScaler
import optuna.integration.lightgbm as lgbm
import optuna
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Number of train samples are",train.shape)
print("Number of test samples are",test.shape)
categorical_features = ['State_Factor', 'building_class', 'facility_type']
numerical_features=train.select_dtypes('number').columns

plt.figure(figsize = (25,11))
sns.heatmap(train.isna().values, cmap = ['#ffd514','#ff355d'], xticklabels=train.columns)
plt.title("Missing values in training Data", size=20)
missing_columns = [col for col in train.columns if train[col].isnull().any()]
missingvalues_count =train.isna().sum()
missingValues_df = pd.DataFrame(missingvalues_count.rename('Null Values Count')).loc[missingvalues_count.ne(0)]
# missingValues_df .style.background_gradient(cmap="Pastel1")

# basic stats of features
# train.describe().style.background_gradient(cmap="Pastel1")

plt.figure(figsize=(15, 7))
plt.subplot(121)
sns.kdeplot(train.site_eui , color = "#ffd514")
plt.subplot(122)
sns.boxplot(train.site_eui , color = "#ff355d")

res = stats.probplot(train['site_eui'], plot=plt)

def kdeplot_features(df_train,df_test, feature, title):
    '''Takes a column from the dataframe and plots the distribution (after count).'''
    
    values_train = df_train[feature].to_numpy()
    values_test = df_test[feature].to_numpy()  
     
    plt.figure(figsize = (18, 3))
    
    sns.kdeplot(values_train, color = '#ffd514')
    sns.kdeplot(values_test, color = '#ff355d')
    
    plt.title(title, fontsize=15)
    plt.legend()
    plt.show();
    
    del values_train , values_test
    gc.collect()
    
def countplot_features(df_train, feature, title):
    '''Takes a column from the dataframe and plots the distribution (after count).'''
    
           
    plt.figure(figsize = (10, 5))
    
    sns.countplot(df_train[feature], color = '#ff355d')
        
    plt.title(title, fontsize=15)    
    plt.show()

# plot distributions of features
# for feature in numerical_features:
#     if feature != "site_eui":
#         kdeplot_features(train,test, feature=feature, title = feature + " distribution")

# plot distributions of categorical features
for feature in categorical_features:
    fig = countplot_features(train, feature=feature, title = "Frequency of "+ feature)

target = train["site_eui"]
train = train.drop(["site_eui","id"],axis =1)
test = test.drop(["id"],axis =1)

# year_built: replace with current year.
train['year_built'] =train['year_built'].replace(np.nan, 2022)
#replacing rest of the values with mean
train['energy_star_rating']=train['energy_star_rating'].replace(np.nan,train['energy_star_rating'].mean())
train['direction_max_wind_speed']= train['direction_max_wind_speed'].replace(np.nan,train['direction_max_wind_speed'].mean())
train['direction_peak_wind_speed']= train['direction_peak_wind_speed'].replace(np.nan,train['direction_peak_wind_speed'].mean())
train['max_wind_speed']=train['max_wind_speed'].replace(np.nan,train['max_wind_speed'].mean())
train['days_with_fog']=train['days_with_fog'].replace(np.nan,train['days_with_fog'].mean())

##for testdata

# year_built: replace with current year.
test['year_built'] =test['year_built'].replace(np.nan, 2022)
#replacing rest of the values with mean
test['energy_star_rating']=test['energy_star_rating'].replace(np.nan,test['energy_star_rating'].mean())
test['direction_max_wind_speed']= test['direction_max_wind_speed'].replace(np.nan,test['direction_max_wind_speed'].mean())
test['direction_peak_wind_speed']= test['direction_peak_wind_speed'].replace(np.nan,test['direction_peak_wind_speed'].mean())
test['max_wind_speed']=test['max_wind_speed'].replace(np.nan,test['max_wind_speed'].mean())
test['days_with_fog']=test['days_with_fog'].replace(np.nan,test['days_with_fog'].mean())

le = LabelEncoder()

train['State_Factor']= le.fit_transform(train['State_Factor']).astype("uint8")
test['State_Factor']= le.fit_transform(test['State_Factor']).astype("uint8")

train['building_class']= le.fit_transform(train['building_class']).astype("uint8")
test['building_class']= le.fit_transform(test['building_class']).astype("uint8")

train['facility_type']= le.fit_transform(train['facility_type']).astype("uint8")
test['facility_type']= le.fit_transform(test['facility_type']).astype("uint8")

# Save train data to W&B Artifacts
train.to_csv("train_features.csv", index = False)

trainnames = copy.deepcopy(train)
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

rkf = RepeatedKFold(n_splits=3, n_repeats=3, random_state=None)

params = {
        "objective": "binary",
        "metric": "binary_error",
        "verbosity": -1,
        "boosting_type": "gbdt",                
        "seed": 42
    }

X = copy.deepcopy(train)  
y = copy.deepcopy(target)

study_tuner = optuna.create_study(direction='minimize')
dtrain = lgbm.Dataset(X, label=y)

# Suppress information only outputs - otherwise optuna is 
# quite verbose, which can be nice, but takes up a lot of space
optuna.logging.set_verbosity(optuna.logging.WARNING) 

# Run optuna LightGBMTunerCV tuning of LightGBM with cross-validation
tuner = lgbm.LightGBMTunerCV(params, 
                            dtrain, 
                            study=study_tuner,
                            verbose_eval=False,                            
                            early_stopping_rounds=250,
                            time_budget=19800, # Time budget of 5 hours, we will not really need it
                            seed = 42,
                            folds=rkf,
                            num_boost_round=10000,
                            callbacks=[lgbm.reset_parameter(learning_rate = [0.005]*200 + [0.001]*9800) ] #[0.1]*5 + [0.05]*15 + [0.01]*45 + 
                           )

tuner.run()

print(tuner.best_params)
# Classification error
print(tuner.best_score)
# Or expressed as accuracy
print(1.0-tuner.best_score)

num_folds = 5
kf = KFold(n_splits = num_folds, random_state = None)
error = 0
models = []
for i, (train_index, val_index) in enumerate(kf.split(train)):
    if i + 1 < num_folds:
        continue
    print(train_index.max(), val_index.min())
    train_X = train[train_index]
    val_X = train[val_index]
    train_y = target[train_index]
    val_y = target[val_index]
    lgb_train = lgb.Dataset(train_X, train_y > 0)
    lgb_eval = lgb.Dataset(val_X, val_y > 0)
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 1.0,
            'bagging_freq' : 0
             
            }
    gbm_class = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=20,
               verbose_eval = 20)
    
    lgb_train = lgb.Dataset(train_X[train_y > 0], train_y[train_y > 0])
    lgb_eval = lgb.Dataset(val_X[val_y > 0] , val_y[val_y > 0])
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 1.0,
            'bagging_freq' : 0
                         }
    gbm_regress = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=20,
               verbose_eval = 20)
#     models.append(gbm)

    y_pred = (gbm_class.predict(val_X, num_iteration=gbm_class.best_iteration) > .5) *\
    (gbm_regress.predict(val_X, num_iteration=gbm_regress.best_iteration))
    error += np.sqrt(mean_squared_error(y_pred, (val_y)))/num_folds
    print(np.sqrt(mean_squared_error(y_pred, (val_y))))
    break
print(error)

feature_imp = pd.DataFrame(sorted(zip(gbm_regress.feature_importance(), trainnames.columns),reverse = True), columns=['Value','Feature'])
feature_imp = feature_imp[feature_imp.Value != 0]
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')

shap_values = shap.TreeExplainer(gbm_regress).shap_values(trainnames)
shap.summary_plot(shap_values, trainnames)

# Prediction on Test Data
res = gbm_regress.predict(test)

# Save results to CSV
sub = pd.read_csv("sample_solution.csv")
sub["site_eui"] = res
sub.to_csv("submission.csv", index = False)