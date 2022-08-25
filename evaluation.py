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
import pickle
from sklearn.metrics import f1_score

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

#Import the already trained model
#the part of Random Forest
rdc_ros_fr = open("trained_models/rdc_ros.pkl", 'rb')
rdc_rus_fr = open("trained_models/rdc_rus.pkl", 'rb')
rdc_smo_fr = open("trained_models/rdc_smo.pkl", 'rb')
rdc_bdlsmo_fr = open("trained_models/rdc_bdlsmo.pkl", 'rb')
rdc_svmsmo_fr = open("trained_models/rdc_svmsmo.pkl", 'rb')
rdc_smote_enn_fr = open("trained_models/rdc_smote_enn.pkl", 'rb')
rdc_smote_tomek_fr = open("trained_models/rdc_smote_tomek.pkl", 'rb')
rdc_nm_fr = open("trained_models/rdc_nm.pkl", 'rb')
rdc_renn_fr = open("trained_models/rdc_renn.pkl", 'rb')
rdc_oss_fr = open("trained_models/rdc_oss.pkl", 'rb')
rdc_ros = joblib.load(rdc_ros_fr)
rdc_rus = joblib.load(rdc_rus_fr)
rdc_smo = joblib.load(rdc_smo_fr)
rdc_bdlsmo = joblib.load(rdc_bdlsmo_fr)
rdc_svmsmo = joblib.load(rdc_svmsmo_fr)
rdc_smote_enn = joblib.load(rdc_smote_enn_fr)
rdc_smote_tomek = joblib.load(rdc_smote_tomek_fr)
rdc_nm = joblib.load(rdc_nm_fr)
rdc_renn = joblib.load(rdc_renn_fr)
rdc_oss = joblib.load(rdc_oss_fr)

#Import the already trained model
#Naive Bayes
gnb_ros_fr = open("trained_models/gnb_ros.pkl", 'rb')
gnb_rus_fr = open("trained_models/gnb_rus.pkl", 'rb')
gnb_smo_fr = open("trained_models/gnb_smo.pkl", 'rb')
gnb_bdlsmo_fr = open("trained_models/gnb_bdlsmo.pkl", 'rb')
gnb_svmsmo_fr = open("trained_models/gnb_svmsmo.pkl", 'rb')
gnb_smote_enn_fr = open("trained_models/gnb_smote_enn.pkl", 'rb')
gnb_smote_tomek_fr = open("trained_models/gnb_smote_tomek.pkl", 'rb')
gnb_nm_fr = open("trained_models/gnb_nm.pkl", 'rb')
gnb_renn_fr = open("trained_models/gnb_renn.pkl", 'rb')
gnb_oss_fr = open("trained_models/gnb_oss.pkl", 'rb')
gnb_ros = joblib.load(gnb_ros_fr)
gnb_rus = joblib.load(gnb_rus_fr)
gnb_smo = joblib.load(gnb_smo_fr)
gnb_bdlsmo = joblib.load(gnb_bdlsmo_fr)
gnb_svmsmo = joblib.load(gnb_svmsmo_fr)
gnb_smote_enn = joblib.load(gnb_smote_enn_fr)
gnb_smote_tomek = joblib.load(gnb_smote_tomek_fr)
gnb_nm = joblib.load(gnb_nm_fr)
gnb_renn = joblib.load(gnb_renn_fr)
gnb_oss = joblib.load(gnb_oss_fr)

#Import the already trained model
#Logistic regression
lr_ros_fr = open("trained_models/lr_ros.pkl", 'rb')
lr_rus_fr = open("trained_models/lr_rus.pkl", 'rb')
lr_smo_fr = open("trained_models/lr_smo.pkl", 'rb')
lr_bdlsmo_fr = open("trained_models/lr_bdlsmo.pkl", 'rb')
lr_svmsmo_fr = open("trained_models/lr_svmsmo.pkl", 'rb')
lr_smote_enn_fr = open("trained_models/lr_smote_enn.pkl", 'rb')
lr_smote_tomek_fr = open("trained_models/lr_smote_tomek.pkl", 'rb')
lr_nm_fr = open("trained_models/lr_nm.pkl", 'rb')
lr_renn_fr = open("trained_models/lr_renn.pkl", 'rb')
lr_oss_fr = open("trained_models/lr_oss.pkl", 'rb')
lr_ros = joblib.load(lr_ros_fr)
lr_rus = joblib.load(lr_rus_fr)
lr_smo = joblib.load(lr_smo_fr)
lr_bdlsmo = joblib.load(lr_bdlsmo_fr)
lr_svmsmo = joblib.load(lr_svmsmo_fr)
lr_smote_enn = joblib.load(lr_smote_enn_fr)
lr_smote_tomek = joblib.load(lr_smote_tomek_fr)
lr_nm = joblib.load(lr_nm_fr)
lr_renn = joblib.load(lr_renn_fr)
lr_oss = joblib.load(lr_oss_fr)

#Import the already trained model
#GradientBoostingClassifier
gbc_ros_fr = open("trained_models/gbc_ros.pkl", 'rb')
gbc_rus_fr = open("trained_models/gbc_rus.pkl", 'rb')
gbc_smo_fr = open("trained_models/gbc_smo.pkl", 'rb')
gbc_bdlsmo_fr = open("trained_models/gbc_bdlsmo.pkl", 'rb')
gbc_svmsmo_fr = open("trained_models/gbc_svmsmo.pkl", 'rb')
gbc_smote_enn_fr = open("trained_models/gbc_smote_enn.pkl", 'rb')
gbc_smote_tomek_fr = open("trained_models/gbc_smote_tomek.pkl", 'rb')
gbc_nm_fr = open("trained_models/gbc_nm.pkl", 'rb')
gbc_renn_fr = open("trained_models/gbc_renn.pkl", 'rb')
gbc_oss_fr = open("trained_models/gbc_oss.pkl", 'rb')
gbc_ros = joblib.load(gbc_ros_fr)
gbc_rus = joblib.load(gbc_rus_fr)
gbc_smo = joblib.load(gbc_smo_fr)
gbc_bdlsmo = joblib.load(gbc_bdlsmo_fr)
gbc_svmsmo = joblib.load(gbc_svmsmo_fr)
gbc_smote_enn = joblib.load(gbc_smote_enn_fr)
gbc_smote_tomek = joblib.load(gbc_smote_tomek_fr)
gbc_nm = joblib.load(gbc_nm_fr)
gbc_renn = joblib.load(gbc_renn_fr)
gbc_oss = joblib.load(gbc_oss_fr)

#Import the already trained model
#XGBClassifier
xgbr_ros_fr = open("trained_models/xgbr_ros.pkl", 'rb')
xgbr_rus_fr = open("trained_models/xgbr_rus.pkl", 'rb')
xgbr_smo_fr = open("trained_models/xgbr_smo.pkl", 'rb')
xgbr_bdlsmo_fr = open("trained_models/xgbr_bdlsmo.pkl", 'rb')
xgbr_svmsmo_fr = open("trained_models/xgbr_svmsmo.pkl", 'rb')
xgbr_smote_enn_fr = open("trained_models/xgbr_smote_enn.pkl", 'rb')
xgbr_smote_tomek_fr = open("trained_models/xgbr_smote_tomek.pkl", 'rb')
xgbr_nm_fr = open("trained_models/xgbr_nm.pkl", 'rb')
xgbr_renn_fr = open("trained_models/xgbr_renn.pkl", 'rb')
xgbr_oss_fr = open("trained_models/xgbr_oss.pkl", 'rb')
xgbr_ros = joblib.load(xgbr_ros_fr)
xgbr_rus = joblib.load(xgbr_rus_fr)
xgbr_smo = joblib.load(xgbr_smo_fr)
xgbr_bdlsmo = joblib.load(xgbr_bdlsmo_fr)
xgbr_svmsmo = joblib.load(xgbr_svmsmo_fr)
xgbr_smote_enn = joblib.load(xgbr_smote_enn_fr)
xgbr_smote_tomek = joblib.load(xgbr_smote_tomek_fr)
xgbr_nm = joblib.load(xgbr_nm_fr)
xgbr_renn = joblib.load(xgbr_renn_fr)
xgbr_oss = joblib.load(xgbr_oss_fr)

data_1718 = pd.read_csv('train_and_test_dataset\combined_data_1718.csv', sep=',')
data_1718

data_19 = pd.read_csv('train_and_test_dataset\combined_data_19.csv', sep=',')
column_ordered_list = [column for column in data_1718]
data_19 = data_19.loc[:, column_ordered_list]
data_19

y_test = data_19['Exclusion'].astype('int')
x_test = data_19.drop(labels = 'Exclusion',axis = 1)

# +
#This part is the image of ROC curve (Random Forest,2019 Dataset)
fig, (ax_1,ax_2) = plt.subplots(2,1)

#the first
metrics.RocCurveDisplay.from_estimator(rdc_ros,x_test,y_test, ax=ax_1, name = 'RandomOverSampler')

#the second
metrics.RocCurveDisplay.from_estimator(rdc_rus,x_test,y_test, ax=ax_1, name = 'RandomUnderSampler')

#the third
metrics.RocCurveDisplay.from_estimator(rdc_smo,x_test,y_test, ax=ax_1, name = 'SMOTE')

#the fourth
metrics.RocCurveDisplay.from_estimator(rdc_bdlsmo,x_test,y_test, ax=ax_1, name = 'BorderlineSMOTE')

#the fifth
metrics.RocCurveDisplay.from_estimator(rdc_svmsmo,x_test,y_test, ax=ax_1, name = 'SVMSMOTE')

#the sixth
metrics.RocCurveDisplay.from_estimator(rdc_smote_enn,x_test,y_test, ax=ax_1, name = 'SMOTEENN')

#the seventh
metrics.RocCurveDisplay.from_estimator(rdc_smote_tomek,x_test,y_test, ax=ax_1, name = 'SMOTETomek')

#the eighth
metrics.RocCurveDisplay.from_estimator(rdc_nm,x_test,y_test, ax=ax_1, name = 'NearMiss')

#the ninth
metrics.RocCurveDisplay.from_estimator(rdc_renn,x_test,y_test, ax=ax_1, name = 'RepeatedEditedNearestNeighbours')

#the tenth
metrics.RocCurveDisplay.from_estimator(rdc_oss,x_test,y_test, ax=ax_1, name = 'OneSidedSelection')

#This part is the image of ROC curve (GradientBoostingClassifier,2019 Dataset)


#the first
metrics.RocCurveDisplay.from_estimator(gbc_ros,x_test,y_test, ax=ax_2, name = 'RandomOverSampler')

# #the second
metrics.RocCurveDisplay.from_estimator(gbc_rus,x_test,y_test, ax=ax_2, name = 'RandomUnderSampler')

# #the third
metrics.RocCurveDisplay.from_estimator(gbc_smo,x_test,y_test, ax=ax_2, name = 'SMOTE')

# #the fourth
metrics.RocCurveDisplay.from_estimator(gbc_bdlsmo,x_test,y_test, ax=ax_2, name = 'BorderlineSMOTE')

# #the fifth
metrics.RocCurveDisplay.from_estimator(gbc_svmsmo,x_test,y_test, ax=ax_2, name = 'SVMSMOTE')

# #the sixth
metrics.RocCurveDisplay.from_estimator(gbc_smote_enn,x_test,y_test, ax=ax_2, name = 'SMOTEENN')

# #the seventh
metrics.RocCurveDisplay.from_estimator(gbc_smote_tomek,x_test,y_test, ax=ax_2, name = 'SMOTETomek')

# #the eighth
metrics.RocCurveDisplay.from_estimator(gbc_nm,x_test,y_test, ax=ax_2, name = 'NearMiss')

# #the ninth
metrics.RocCurveDisplay.from_estimator(gbc_renn,x_test,y_test, ax=ax_2, name = 'RepeatedEditedNearestNeighbours')

# #the tenth
metrics.RocCurveDisplay.from_estimator(gbc_oss,x_test,y_test, ax=ax_2, name = 'OneSidedSelection')


fig.set_figheight(12)
fig.set_figwidth(8)
fig.suptitle('The ROC Curves of Random Forest(top) and GradientBoosting(bottom) with 2019 dataset')
plt.savefig("metrics_pictures/Random_Forest_and_GB_ROC.jpg")
plt.show()

# +
#This part is the image of ROC curve (Naive Bayes,2019 Dataset)
fig, (ax_1,ax_2) = plt.subplots(2,1)

#the first
metrics.RocCurveDisplay.from_estimator(gnb_ros,x_test,y_test, ax=ax_1, name = 'RandomOverSampler')

# #the second
metrics.RocCurveDisplay.from_estimator(gnb_rus,x_test,y_test, ax=ax_1, name = 'RandomUnderSampler')

# #the third
metrics.RocCurveDisplay.from_estimator(gnb_smo,x_test,y_test, ax=ax_1, name = 'SMOTE')

# #the fourth
metrics.RocCurveDisplay.from_estimator(gnb_bdlsmo,x_test,y_test, ax=ax_1, name = 'BorderlineSMOTE')

# #the fifth
metrics.RocCurveDisplay.from_estimator(gnb_svmsmo,x_test,y_test, ax=ax_1, name = 'SVMSMOTE')

# #the sixth
metrics.RocCurveDisplay.from_estimator(gnb_smote_enn,x_test,y_test, ax=ax_1, name = 'SMOTEENN')

# #the seventh
metrics.RocCurveDisplay.from_estimator(gnb_smote_tomek,x_test,y_test, ax=ax_1, name = 'SMOTETomek')

# #the eighth
metrics.RocCurveDisplay.from_estimator(gnb_nm,x_test,y_test, ax=ax_1, name = 'NearMiss')

# #the ninth
metrics.RocCurveDisplay.from_estimator(gnb_renn,x_test,y_test, ax=ax_1, name = 'RepeatedEditedNearestNeighbours')

# #the tenth
metrics.RocCurveDisplay.from_estimator(gnb_oss,x_test,y_test, ax=ax_1, name = 'OneSidedSelection')


#This part is the image of ROC curve (Logistic Regression,2019 Dataset)


#the first
metrics.RocCurveDisplay.from_estimator(lr_ros,x_test,y_test, ax=ax_2, name = 'RandomOverSampler')

# #the second
metrics.RocCurveDisplay.from_estimator(lr_rus,x_test,y_test, ax=ax_2, name = 'RandomUnderSampler')

# #the third
metrics.RocCurveDisplay.from_estimator(lr_smo,x_test,y_test, ax=ax_2, name = 'SMOTE')

# #the fourth
metrics.RocCurveDisplay.from_estimator(lr_bdlsmo,x_test,y_test, ax=ax_2, name = 'BorderlineSMOTE')

# #the fifth
metrics.RocCurveDisplay.from_estimator(lr_svmsmo,x_test,y_test, ax=ax_2, name = 'SVMSMOTE')

# #the sixth
metrics.RocCurveDisplay.from_estimator(lr_smote_enn,x_test,y_test, ax=ax_2, name = 'SMOTEENN')

# #the seventh
metrics.RocCurveDisplay.from_estimator(lr_smote_tomek,x_test,y_test, ax=ax_2, name = 'SMOTETomek')

# #the eighth
metrics.RocCurveDisplay.from_estimator(lr_nm,x_test,y_test, ax=ax_2, name = 'NearMiss')

# #the ninth
metrics.RocCurveDisplay.from_estimator(lr_renn,x_test,y_test, ax=ax_2, name = 'RepeatedEditedNearestNeighbours')

# #the tenth
metrics.RocCurveDisplay.from_estimator(lr_oss,x_test,y_test, ax=ax_2, name = 'OneSidedSelection')

fig.set_figheight(12)
fig.set_figwidth(8)
fig.suptitle('The ROC Curves of Naive Bayes(top) and Logistic Regression(bottom) with 2019 dataset')
plt.savefig("metrics_pictures/NB_and_Logistic_Regression_ROC.jpg")
plt.show()

# +
#This part is the image of ROC curve (XGBClassifier,2019 Dataset)
fig, ax = plt.subplots()

#the first
metrics.RocCurveDisplay.from_estimator(xgbr_ros,x_test,y_test, ax=ax, name = 'RandomOverSampler')

# #the second
metrics.RocCurveDisplay.from_estimator(xgbr_rus,x_test,y_test, ax=ax, name = 'RandomUnderSampler')

# #the third
metrics.RocCurveDisplay.from_estimator(xgbr_smo,x_test,y_test, ax=ax, name = 'SMOTE')

# #the fourth
metrics.RocCurveDisplay.from_estimator(xgbr_bdlsmo,x_test,y_test, ax=ax, name = 'BorderlineSMOTE')

# #the fifth
metrics.RocCurveDisplay.from_estimator(xgbr_svmsmo,x_test,y_test, ax=ax, name = 'SVMSMOTE')

# #the sixth
metrics.RocCurveDisplay.from_estimator(xgbr_smote_enn,x_test,y_test, ax=ax, name = 'SMOTEENN')

# #the seventh
metrics.RocCurveDisplay.from_estimator(xgbr_smote_tomek,x_test,y_test, ax=ax, name = 'SMOTETomek')

# #the eighth
metrics.RocCurveDisplay.from_estimator(xgbr_nm,x_test,y_test, ax=ax, name = 'NearMiss')

# #the ninth
metrics.RocCurveDisplay.from_estimator(xgbr_renn,x_test,y_test, ax=ax, name = 'RepeatedEditedNearestNeighbours')

# #the tenth
metrics.RocCurveDisplay.from_estimator(xgbr_oss,x_test,y_test, ax=ax, name = 'OneSidedSelection')

fig.set_figheight(7)
fig.set_figwidth(7)
fig.suptitle('The ROC Curves of XGBoost with 2019 dataset')
plt.savefig("metrics_pictures/XGBr_ROC.jpg")
plt.show()

# +
#These are the predicted values for the test set under Naive Bayes

y_gnb_ros_pred = gnb_ros.predict(x_test)
y_gnb_rus_pred = gnb_rus.predict(x_test)
y_gnb_smo_pred = gnb_smo.predict(x_test)
y_gnb_bdlsmo_pred = gnb_bdlsmo.predict(x_test)
y_gnb_svmsmo_pred = gnb_svmsmo.predict(x_test)
y_gnb_smote_enn_pred = gnb_smote_enn.predict(x_test)
y_gnb_smote_tomek_pred = gnb_smote_tomek.predict(x_test)
y_gnb_nm_pred = gnb_nm.predict(x_test)
y_gnb_renn_pred = gnb_renn.predict(x_test)
y_gnb_oss_pred = gnb_oss.predict(x_test)
# -

#Generate the corresponding F1-score value
mcc_gnb_ros = f1_score(y_test, y_gnb_ros_pred,pos_label=0)
mcc_gnb_rus = f1_score(y_test, y_gnb_rus_pred,pos_label=0)
mcc_gnb_smo = f1_score(y_test, y_gnb_smo_pred,pos_label=0)
mcc_gnb_bdlsmo = f1_score(y_test, y_gnb_bdlsmo_pred,pos_label=0)
mcc_gnb_svmsmo = f1_score(y_test, y_gnb_svmsmo_pred,pos_label=0)
mcc_gnb_smote_enn = f1_score(y_test, y_gnb_smote_enn_pred,pos_label=0)
mcc_gnb_smote_tomek = f1_score(y_test, y_gnb_smote_tomek_pred,pos_label=0)
mcc_gnb_nm = f1_score(y_test, y_gnb_nm_pred,pos_label=0)
mcc_gnb_renn = f1_score(y_test, y_gnb_renn_pred,pos_label=0)
mcc_gnb_oss = f1_score(y_test, y_gnb_oss_pred,pos_label=0)
list_gnb = [mcc_gnb_ros,mcc_gnb_rus,mcc_gnb_smo,mcc_gnb_bdlsmo,mcc_gnb_svmsmo,mcc_gnb_smote_enn,mcc_gnb_smote_tomek,mcc_gnb_nm,mcc_gnb_renn,mcc_gnb_oss]
list_gnb

# +
#These are the predicted values for the test set, random forests

y_rdc_ros_pred = rdc_ros.predict(x_test)
y_rdc_rus_pred = rdc_rus.predict(x_test)
y_rdc_smo_pred = rdc_smo.predict(x_test)
y_rdc_bdlsmo_pred = rdc_bdlsmo.predict(x_test)
y_rdc_svmsmo_pred = rdc_svmsmo.predict(x_test)
y_rdc_smote_enn_pred = rdc_smote_enn.predict(x_test)
y_rdc_smote_tomek_pred = rdc_smote_tomek.predict(x_test)
y_rdc_nm_pred = rdc_nm.predict(x_test)
y_rdc_renn_pred = rdc_renn.predict(x_test)
y_rdc_oss_pred = rdc_oss.predict(x_test)
# -

#Generate the corresponding F1-score value, Random Forest
mcc_rdc_ros = f1_score(y_test.array, y_rdc_ros_pred,pos_label=0)
mcc_rdc_rus = f1_score(y_test.array, y_rdc_rus_pred,pos_label=0)
mcc_rdc_smo = f1_score(y_test.array, y_rdc_smo_pred,pos_label=0)
mcc_rdc_bdlsmo = f1_score(y_test.array, y_rdc_bdlsmo_pred,pos_label=0)
mcc_rdc_svmsmo = f1_score(y_test.array, y_rdc_svmsmo_pred,pos_label=0)
mcc_rdc_smote_enn = f1_score(y_test.array, y_rdc_smote_enn_pred,pos_label=0)
mcc_rdc_smote_tomek = f1_score(y_test.array, y_rdc_smote_tomek_pred,pos_label=0)
mcc_rdc_nm = f1_score(y_test.array, y_rdc_nm_pred,pos_label=0)
mcc_rdc_renn = f1_score(y_test.array, y_rdc_renn_pred,pos_label=0)
mcc_rdc_oss = f1_score(y_test.array, y_rdc_oss_pred,pos_label=0)
list_rdc = [mcc_rdc_ros,mcc_rdc_rus,mcc_rdc_smo,mcc_rdc_bdlsmo,mcc_rdc_svmsmo,mcc_rdc_smote_enn,mcc_rdc_smote_tomek,mcc_rdc_nm,mcc_rdc_renn,mcc_rdc_oss]
list_rdc

# +
#These are the predicted values for the test set，Logistic Regression

y_lr_ros_pred = lr_ros.predict(x_test)
y_lr_rus_pred = lr_rus.predict(x_test)
y_lr_smo_pred = lr_smo.predict(x_test)
y_lr_bdlsmo_pred = lr_bdlsmo.predict(x_test)
y_lr_svmsmo_pred = lr_svmsmo.predict(x_test)
y_lr_smote_enn_pred = lr_smote_enn.predict(x_test)
y_lr_smote_tomek_pred = lr_smote_tomek.predict(x_test)
y_lr_nm_pred = lr_nm.predict(x_test)
y_lr_renn_pred = lr_renn.predict(x_test)
y_lr_oss_pred = lr_oss.predict(x_test)
# -

#Generate the corresponding F1-score value，Logistic Regression
mcc_lr_ros = f1_score(y_test, y_lr_ros_pred,pos_label=0)
mcc_lr_rus = f1_score(y_test, y_lr_rus_pred,pos_label=0)
mcc_lr_smo = f1_score(y_test, y_lr_smo_pred,pos_label=0)
mcc_lr_bdlsmo = f1_score(y_test, y_lr_bdlsmo_pred,pos_label=0)
mcc_lr_svmsmo = f1_score(y_test, y_lr_svmsmo_pred,pos_label=0)
mcc_lr_smote_enn = f1_score(y_test, y_lr_smote_enn_pred,pos_label=0)
mcc_lr_smote_tomek = f1_score(y_test, y_lr_smote_tomek_pred,pos_label=0)
mcc_lr_nm = f1_score(y_test, y_lr_nm_pred,pos_label=0)
mcc_lr_renn = f1_score(y_test, y_lr_renn_pred,pos_label=0)
mcc_lr_oss = f1_score(y_test, y_lr_oss_pred,pos_label=0)
list_lr = [mcc_lr_ros,mcc_lr_rus,mcc_lr_smo,mcc_lr_bdlsmo,mcc_lr_svmsmo,mcc_lr_smote_enn,mcc_lr_smote_tomek,mcc_lr_nm,mcc_lr_renn,mcc_lr_oss]
list_lr

# +
#These are the predicted values for the test set，GBM

y_gbc_ros_pred = gbc_ros.predict(x_test)
y_gbc_rus_pred = gbc_rus.predict(x_test)
y_gbc_smo_pred = gbc_smo.predict(x_test)
y_gbc_bdlsmo_pred = gbc_bdlsmo.predict(x_test)
y_gbc_svmsmo_pred = gbc_svmsmo.predict(x_test)
y_gbc_smote_enn_pred = gbc_smote_enn.predict(x_test)
y_gbc_smote_tomek_pred = gbc_smote_tomek.predict(x_test)
y_gbc_nm_pred = gbc_nm.predict(x_test)
y_gbc_renn_pred = gbc_renn.predict(x_test)
y_gbc_oss_pred = gbc_oss.predict(x_test)
# -

#Generate the corresponding F1-score value，GBM
mcc_gbc_ros = f1_score(y_test, y_gbc_ros_pred,pos_label=0)
mcc_gbc_rus = f1_score(y_test, y_gbc_rus_pred,pos_label=0)
mcc_gbc_smo = f1_score(y_test, y_gbc_smo_pred,pos_label=0)
mcc_gbc_bdlsmo = f1_score(y_test, y_gbc_bdlsmo_pred,pos_label=0)
mcc_gbc_svmsmo = f1_score(y_test, y_gbc_svmsmo_pred,pos_label=0)
mcc_gbc_smote_enn = f1_score(y_test, y_gbc_smote_enn_pred,pos_label=0)
mcc_gbc_smote_tomek = f1_score(y_test, y_gbc_smote_tomek_pred,pos_label=0)
mcc_gbc_nm = f1_score(y_test, y_gbc_nm_pred,pos_label=0)
mcc_gbc_renn = f1_score(y_test, y_gbc_renn_pred,pos_label=0)
mcc_gbc_oss = f1_score(y_test, y_gbc_oss_pred,pos_label=0)
list_gbc = [mcc_gbc_ros,mcc_gbc_rus,mcc_gbc_smo,mcc_gbc_bdlsmo,mcc_gbc_svmsmo,mcc_gbc_smote_enn,mcc_gbc_smote_tomek,mcc_gbc_nm,mcc_gbc_renn,mcc_gbc_oss]
list_gbc

# +
#These are the predicted values for the test set，XGBoost

y_xgbr_ros_pred = xgbr_ros.predict(x_test)
y_xgbr_rus_pred = xgbr_rus.predict(x_test)
y_xgbr_smo_pred = xgbr_smo.predict(x_test)
y_xgbr_bdlsmo_pred = xgbr_bdlsmo.predict(x_test)
y_xgbr_svmsmo_pred = xgbr_svmsmo.predict(x_test)
y_xgbr_smote_enn_pred = xgbr_smote_enn.predict(x_test)
y_xgbr_smote_tomek_pred = xgbr_smote_tomek.predict(x_test)
y_xgbr_nm_pred = xgbr_nm.predict(x_test)
y_xgbr_renn_pred = xgbr_renn.predict(x_test)
y_xgbr_oss_pred = xgbr_oss.predict(x_test)
# -

#Generate the corresponding F1-score value，XGBoost
mcc_xgbr_ros = f1_score(y_test, y_xgbr_ros_pred,pos_label=0)
mcc_xgbr_rus = f1_score(y_test, y_xgbr_rus_pred,pos_label=0)
mcc_xgbr_smo = f1_score(y_test, y_xgbr_smo_pred,pos_label=0)
mcc_xgbr_bdlsmo = f1_score(y_test, y_xgbr_bdlsmo_pred,pos_label=0)
mcc_xgbr_svmsmo = f1_score(y_test, y_xgbr_svmsmo_pred,pos_label=0)
mcc_xgbr_smote_enn = f1_score(y_test, y_xgbr_smote_enn_pred,pos_label=0)
mcc_xgbr_smote_tomek = f1_score(y_test, y_xgbr_smote_tomek_pred,pos_label=0)
mcc_xgbr_nm = f1_score(y_test, y_xgbr_nm_pred,pos_label=0)
mcc_xgbr_renn = f1_score(y_test, y_xgbr_renn_pred,pos_label=0)
mcc_xgbr_oss = f1_score(y_test, y_xgbr_oss_pred,pos_label=0)
list_xgbr = [mcc_xgbr_ros,mcc_xgbr_rus,mcc_xgbr_smo,mcc_xgbr_bdlsmo,mcc_xgbr_svmsmo,mcc_xgbr_smote_enn,mcc_xgbr_smote_tomek,mcc_xgbr_nm,mcc_xgbr_renn,mcc_xgbr_oss]
list_xgbr


