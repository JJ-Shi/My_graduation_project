# -*- coding: utf-8 -*-
# +
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
import joblib

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
# -

#log the training dataset
data_1718 = pd.read_csv('train_and_test_dataset\combined_data_1718.csv', sep=',')
data_1718

#log the testing dataset
data_19 = pd.read_csv('train_and_test_dataset\combined_data_19.csv', sep=',')
column_ordered_list = [column for column in data_1718]
data_19 = data_19.loc[:, column_ordered_list]
data_19

y_test = data_19['Exclusion'].astype('int')
x_test = data_19.drop(labels = 'Exclusion',axis = 1)

#The first kind of resampled dataset
x_resampled_ros = pd.read_csv('resampled_datasets/x_resampled_ros.csv',sep=',')
y_resampled_ros = pd.read_csv('resampled_datasets/y_resampled_ros.csv',sep=',')

#The second kind of resampled dataset
x_resampled_rus = pd.read_csv('resampled_datasets/x_resampled_rus.csv',sep=',')
y_resampled_rus = pd.read_csv('resampled_datasets/y_resampled_rus.csv',sep=',')

The third kind of resampled dataset
x_resampled_smo = pd.read_csv('resampled_datasets/x_resampled_smo.csv',sep=',')
y_resampled_smo = pd.read_csv('resampled_datasets/y_resampled_smo.csv',sep=',')

#The fourth kind of resampled dataset
x_resampled_bdlsmo = pd.read_csv('resampled_datasets/x_resampled_bdlsmo.csv',sep=',')
y_resampled_bdlsmo = pd.read_csv('resampled_datasets/y_resampled_bdlsmo.csv',sep=',')

#The fifth kind of resampled dataset
x_resampled_svmsmo = pd.read_csv('resampled_datasets/x_resampled_svmsmo.csv',sep=',')
y_resampled_svmsmo = pd.read_csv('resampled_datasets/y_resampled_svmsmo.csv',sep=',')

#The sixth kind of resampled dataset
x_resampled_smote_enn = pd.read_csv('resampled_datasets/x_resampled_smote_enn.csv',sep=',')
y_resampled_smote_enn = pd.read_csv('resampled_datasets/y_resampled_smote_enn.csv',sep=',')

#the seventh kind of resampled dataset
x_resampled_smote_tomek = pd.read_csv('resampled_datasets/x_resampled_smote_tomek.csv',sep=',')
y_resampled_smote_tomek = pd.read_csv('resampled_datasets/y_resampled_smote_tomek.csv',sep=',')

#The eighth kind of resampled dataset
x_resampled_nm = pd.read_csv('resampled_datasets/x_resampled_nm.csv',sep=',')
y_resampled_nm = pd.read_csv('resampled_datasets/y_resampled_nm.csv',sep=',')

#The ninth kind of resampled dataset
x_resampled_renn = pd.read_csv('resampled_datasets/x_resampled_renn.csv',sep=',')
y_resampled_renn = pd.read_csv('resampled_datasets/y_resampled_renn.csv',sep=',')

#The tenth kind of resampled dataset
x_resampled_oss = pd.read_csv('resampled_datasets/x_resampled_oss.csv',sep=',')
y_resampled_oss = pd.read_csv('resampled_datasets/y_resampled_oss.csv',sep=',')

#GBM trained on the first dataset
gbc_ros = GradientBoostingClassifier()
gbc_ros = gbc_ros.fit(x_resampled_ros, y_resampled_ros)
joblib.dump(gbc_ros, 'trained_models/gbc_ros.pkl')

#GBM trained on the second dataset
gbc_rus = GradientBoostingClassifier()
gbc_rus = gbc_rus.fit(x_resampled_rus, y_resampled_rus)
joblib.dump(gbc_rus, 'trained_models/gbc_rus.pkl')

GBM trained on the third dataset
gbc_smo = GradientBoostingClassifier()
gbc_smo = gbc_smo.fit(x_resampled_smo, y_resampled_smo)
joblib.dump(gbc_smo, 'trained_models/gbc_smo.pkl')

#GBM trained on the fourth dataset
gbc_bdlsmo = GradientBoostingClassifier()
gbc_bdlsmo = gbc_bdlsmo.fit(x_resampled_bdlsmo, y_resampled_bdlsmo)
joblib.dump(gbc_bdlsmo, 'trained_models/gbc_bdlsmo.pkl')

#GBM trained on the fifth dataset
gbc_svmsmo = GradientBoostingClassifier()
gbc_svmsmo = gbc_svmsmo.fit(x_resampled_svmsmo, y_resampled_svmsmo)
joblib.dump(gbc_svmsmo, 'trained_models/gbc_svmsmo.pkl')

#GBM trained on the sixth dataset
gbc_smote_enn = GradientBoostingClassifier()
gbc_smote_enn = gbc_smote_enn.fit(x_resampled_smote_enn, y_resampled_smote_enn)
joblib.dump(gbc_smote_enn, 'trained_models/gbc_smote_enn.pkl')

#GBM trained on the seventh dataset
gbc_smote_tomek = GradientBoostingClassifier()
gbc_smote_tomek = gbc_smote_tomek.fit(x_resampled_smote_tomek, y_resampled_smote_tomek)
joblib.dump(gbc_smote_tomek, 'trained_models/gbc_smote_tomek.pkl')

#GBM trained on the eighth dataset
gbc_nm = GradientBoostingClassifier()
gbc_nm = gbc_nm.fit(x_resampled_nm, y_resampled_nm)
joblib.dump(gbc_nm, 'trained_models/gbc_nm.pkl')

#GBM trained on the ninth dataset
gbc_renn = GradientBoostingClassifier()
gbc_renn = gbc_renn.fit(x_resampled_renn, y_resampled_renn)
joblib.dump(gbc_renn, 'trained_models/gbc_renn.pkl')

#GBM trained on the tenth dataset
gbc_oss = GradientBoostingClassifier()
gbc_oss = gbc_oss.fit(x_resampled_oss, y_resampled_oss)
joblib.dump(gbc_oss, 'trained_models/gbc_oss.pkl')

#XGBoost trained on the first dataset
xgbr_ros = xgboost.XGBClassifier()
xgbr_ros = xgbr_ros.fit(x_resampled_ros, y_resampled_ros)
joblib.dump(xgbr_ros, 'trained_models/xgbr_ros.pkl')

#XGBoost trained on the second dataset
xgbr_rus = xgboost.XGBClassifier()
xgbr_rus = xgbr_rus.fit(x_resampled_rus, y_resampled_rus)
joblib.dump(xgbr_rus, 'trained_models/xgbr_rus.pkl')

#XGBoost trained on the third dataset
xgbr_smo = xgboost.XGBClassifier()
xgbr_smo = xgbr_smo.fit(x_resampled_smo, y_resampled_smo)
joblib.dump(xgbr_smo, 'trained_models/xgbr_smo.pkl')

#XGBoost trained on the fourth dataset
xgbr_bdlsmo = xgboost.XGBClassifier()
xgbr_bdlsmo = xgbr_bdlsmo.fit(x_resampled_bdlsmo, y_resampled_bdlsmo)
joblib.dump(xgbr_bdlsmo, 'trained_models/xgbr_bdlsmo.pkl')

#XGBoost trained on the fifth dataset
xgbr_svmsmo = xgboost.XGBClassifier()
xgbr_svmsmo = xgbr_svmsmo.fit(x_resampled_svmsmo, y_resampled_svmsmo)
joblib.dump(xgbr_svmsmo, 'trained_models/xgbr_svmsmo.pkl')

#XGBoost trained on the sixth dataset
xgbr_smote_enn = xgboost.XGBClassifier()
xgbr_smote_enn = xgbr_smote_enn.fit(x_resampled_smote_enn, y_resampled_smote_enn)
joblib.dump(xgbr_smote_enn, 'trained_models/xgbr_smote_enn.pkl')

#XGBoost trained on the seventh dataset
xgbr_smote_tomek = xgboost.XGBClassifier()
xgbr_smote_tomek = xgbr_smote_tomek.fit(x_resampled_smote_tomek, y_resampled_smote_tomek)
joblib.dump(xgbr_smote_tomek, 'trained_models/xgbr_smote_tomek.pkl')

#XGBoost trained on the eighth dataset
xgbr_nm = xgboost.XGBClassifier()
xgbr_nm = xgbr_nm.fit(x_resampled_nm, y_resampled_nm)
joblib.dump(xgbr_nm, 'trained_models/xgbr_nm.pkl')

#XGBoost trained on the ninth dataset
xgbr_renn = xgboost.XGBClassifier()
xgbr_renn = xgbr_renn.fit(x_resampled_renn, y_resampled_renn)
joblib.dump(xgbr_renn, 'trained_models/xgbr_renn.pkl')

#XGBoost trained on the tenth dataset
xgbr_oss = xgboost.XGBClassifier()
xgbr_oss = xgbr_oss.fit(x_resampled_oss, y_resampled_oss)
joblib.dump(xgbr_oss, 'trained_models/xgbr_oss.pkl')


