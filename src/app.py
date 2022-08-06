
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

#if value =0 and diabetic, set the mean of diabetic, else if value=0 and not diabetic, set the mean of not diabetic, 
# rest of the case use the data value

def set_value (data_value, outcome_value,mean_nodiab,mean_diab):
    if (outcome_value == 0 and data_value==0):
        return mean_nodiab
    elif (outcome_value ==1 and data_value ==0):
        return mean_diab
    else:
        return data_value

def set_use_mean (name_col):
    #calc the mean for diabetic and not diabetic that the data is not 0
    meanNoDiab = df[(df[name_col]>0) & (df['Outcome']==0)][name_col].mean()
    meanDiab = df[(df[name_col]>0) & (df['Outcome']==1)][name_col].mean()

    df[name_col] = df.apply(lambda x: set_value(x[name_col], x['Outcome'],meanNoDiab,meanDiab), axis=1)

set_use_mean('Insulin')
set_use_mean('Glucose')
set_use_mean('BloodPressure')
set_use_mean('SkinThickness')
set_use_mean('BMI')


 
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

#exclude pregnancy and BloodPressure to feature model
 
#X = df_raw[list(df_raw.columns[1:8])]
X = df[['Glucose','SkinThickness','Insulin','BMI','Age','DiabetesPedigreeFunction']] 
y = df[['Outcome']]

#Separate features from target
X=df.drop('Outcome',axis='columns')
Y=df["Outcome"]

#Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(X)


#use the model save with new data to predicts

filename = '../models/finalized_model1.sav' #use absolute path
loaded_model = pickle.load(open(filename, 'rb'))

#Predict using the model 
Glucose=120
SkinThickness=23
Insulin=215
BMI=29
Age=24
DiabetesPedigreeFunction=0.520


#predigo el target para los valores seteados con modelo
print('Predicted Diabetic : \n', loaded_model.predict([[Glucose,SkinThickness,Insulin,BMI,Age,DiabetesPedigreeFunction]]))

Glucose=134
SkinThickness=30
Insulin=74
BMI=34
Age=24
DiabetesPedigreeFunction=0.75

#predigo el target para los valores seteados con modelo
print('Predicted Diabetic : \n', loaded_model.predict([[Glucose,SkinThickness,Insulin,BMI,Age,DiabetesPedigreeFunction]]))





