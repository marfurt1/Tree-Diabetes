!pip install pandas
!pip install matplotlib
!pip install sklearn
!pip install seaborn
!pip install folium
!pip install statsmodels
!pip install plotly
!pip install DecisionTreeClassifier
!pip install pickle

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle

url = 'https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
df = pd.read_csv(url, header=0, sep=",")

# Clean the  DataSet

#insulina
df_0=df[(df['Outcome']==0) & (df["Insulin"] > 0)]
insuline_mean_0=df_0['Insulin'].mean()

df_no0=df[(df['Outcome']!=0) & (df["Insulin"] > 0)]
insuline_mean_no0=df_no0['Insulin'].mean()

def insulina(insulin_value, outcome_value, insuline_mean_0,insuline_mean_no0):
    if outcome_value==0 and insulin_value==0:
        return insuline_mean_0
    elif outcome_value==1 and insulin_value==0:
        return insuline_mean_no0
    else:
        return insulin_value

df['Insulin'] = df.apply(lambda x: insulina(x['Insulin'], x['Outcome'],insuline_mean_0,insuline_mean_no0), axis=1)

#bmi
df_0=df[(df['Outcome']==0) & (df["BMI"] > 0)]
bmi_mean_0=df_0['BMI'].mean()

df_no0=df[(df['Outcome']!=0) & (df["BMI"] > 0)]
bmi_mean_no0=df_no0['BMI'].mean()

def BMI_fun(bmi_value, outcome_value, bmi_mean_0,bmi_mean_no0):
    if outcome_value==0 and bmi_value==0:
        return bmi_mean_0
    elif outcome_value==1 and bmi_value==0:
        return bmi_mean_no0
    else:
        return bmi_value

df['BMI'] = df.apply(lambda x: BMI_fun(x['BMI'], x['Outcome'],bmi_mean_0,bmi_mean_no0), axis=1)

#Glucose
df_0=df[(df['Outcome']==0) & (df["Glucose"] > 0)]
glucose_mean_0=df_0['Glucose'].mean()

df_no0=df[(df['Outcome']!=0) & (df["Glucose"] > 0)]
glucose_mean_no0=df_no0['Glucose'].mean()

def glucose_fun(glucose_value, outcome_value, glucose_mean_0,glucose_mean_no0):
    if outcome_value==0 and glucose_value==0:
        return glucose_mean_0
    elif outcome_value==1 and glucose_value==0:
        return glucose_mean_no0
    else:
        return glucose_value

df['Glucose'] = df.apply(lambda x: glucose_fun(x['Glucose'], x['Outcome'],glucose_mean_0,glucose_mean_no0), axis=1)

#BloodPressure
df_0=df[(df['Outcome']==0) & (df["BloodPressure"] > 0)]
BloodPressure_mean_0=df_0['BloodPressure'].mean()

df_no0=df[(df['Outcome']!=0) & (df["BloodPressure"] > 0)]
BloodPressure_mean_no0=df_no0['BloodPressure'].mean()

def BloodPressure_fun(BloodPressure_value, outcome_value, BloodPressure_mean_0,BloodPressure_mean_no0):
    if outcome_value==0 and BloodPressure_value==0:
        return BloodPressure_mean_0
    elif outcome_value==1 and BloodPressure_value==0:
        return BloodPressure_mean_no0
    else:
        return BloodPressure_value

df['BloodPressure'] = df.apply(lambda x: BloodPressure_fun(x['BloodPressure'], x['Outcome'],BloodPressure_mean_0,BloodPressure_mean_no0), axis=1)

#SkinThickness
df_0=df[(df['Outcome']==0) & (df["SkinThickness"] > 0)]
SkinThickness_mean_0=df_0['SkinThickness'].mean()

df_no0=df[(df['Outcome']!=0) & (df["SkinThickness"] > 0)]
SkinThickness_mean_no0=df_no0['SkinThickness'].mean()

def SkinThickness_fun(SkinThickness_value, outcome_value, SkinThickness_mean_0,SkinThickness_mean_no0):
    if outcome_value==0 and SkinThickness_value==0:
        return SkinThickness_mean_0
    elif outcome_value==1 and SkinThickness_value==0:
        return SkinThickness_mean_no0
    else:
        return SkinThickness_value

df['SkinThickness'] = df.apply(lambda x: SkinThickness_fun(x['SkinThickness'], x['Outcome'],SkinThickness_mean_0,SkinThickness_mean_no0), axis=1)

#Resample the data to make a more balanced dataset
from sklearn.utils import resample
df_majority = df[(df["Outcome"]==0)]
df_minority = df[(df["Outcome"]==1)]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=500,random_state=42)
df = pd.concat([df_minority_upsampled,df_majority])

#Eliminating Outliers with IQR Method
q1 = df['Insulin'].quantile(0.25)
q3 = df['Insulin'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR

df = df[(df['Insulin'] > lower_limit) & (df['Insulin'] < upper_limit)]

q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR

df = df[(df['Age'] > lower_limit) & (df['Age'] < upper_limit)]

q1 = df['Glucose'].quantile(0.25)
q3 = df['Glucose'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR

df = df[(df['Glucose'] > lower_limit) & (df['Glucose'] < upper_limit)]

q1 = df['BMI'].quantile(0.25)
q3 = df['BMI'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR

df = df[(df['BMI'] > lower_limit) & (df['BMI'] < upper_limit)]

q1 = df['Pregnancies'].quantile(0.25)
q3 = df['Pregnancies'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR

df = df[(df['Pregnancies'] > lower_limit) & (df['Pregnancies'] < upper_limit)]

q1 = df['DiabetesPedigreeFunction'].quantile(0.25)
q3 = df['DiabetesPedigreeFunction'].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 1.5 * IQR
upper_limit = q3 + 1.5 * IQR

df = df[(df['DiabetesPedigreeFunction'] > lower_limit) & (df['DiabetesPedigreeFunction'] < upper_limit)]

#Separate features from target
X=df.drop('Outcome',axis='columns')
Y=df["Outcome"]

#Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(X)




