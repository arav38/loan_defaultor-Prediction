
#importing the libraries

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

#loading the data in system
XYZ_DF=pd.read_csv(r"C:\akshay\Data Science Final Project\XYZCorp_LendingData (1).txt",header=0 ,
                      delimiter="\t", low_memory=False)

# checking the number of variable and observation using shape command
XYZ_DF.shape   #(855969, 73)
# checking the data
XYZ_DF.head(

#create copy of data
XYZ_DF_rev=pd.DataFrame.copy(XYZ_DF)
XYZ_DF_rev.shape #(855969, 73)
		  

#Feature Selection
# Out of 73 , few variables are not helpful or impactful in order to build a predictive model, hence dropping.

XYZ_DF_rev.drop(['id','member_id','funded_amnt_inv','grade','emp_title','pymnt_plan','desc','title','addr_state',
            'inq_last_6mths','mths_since_last_record','initial_list_status','mths_since_last_major_derog','policy_code',
            'dti_joint','verification_status_joint','tot_coll_amt','tot_cur_bal','open_acc_6m','open_il_6m','open_il_12m'
            ,'open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m',
            'max_bal_bc','all_util','inq_fi','total_cu_tl','inq_last_12m'],axis=1,inplace=True)
	    

# checking the number of variable and observation using shape command
XYZ_DF_rev.shape #(855969, 41)
# checking the data
#XYZ_DF_rev.head()	   

# Checking if missing values are present and datatype of each variable.
XYZ_DF_rev.isnull().sum()
print(XYZ_DF_rev.dtypes)


# Imputing missing data for Numerical with mean value / Zeros 
XYZ_DF_rev['annual_inc_joint'].fillna(0,inplace=True)

colname2=['mths_since_last_delinq','revol_util','collections_12_mths_ex_med',
          'total_rev_hi_lim']
for x in colname2[:]:
    XYZ_DF_rev[x].fillna(XYZ_DF_rev[x].mean(),inplace=True)
    
XYZ_DF_rev.isnull().sum()
XYZ_DF_rev.shape 

# Imputing missing data for Numerical with mean value / Zeros 
XYZ_DF_rev['annual_inc_joint'].fillna(0,inplace=True)

colname2=['mths_since_last_delinq','revol_util','collections_12_mths_ex_med',
          'total_rev_hi_lim']
for x in colname2[:]:
    XYZ_DF_rev[x].fillna(XYZ_DF_rev[x].mean(),inplace=True)
    
XYZ_DF_rev.isnull().sum()
XYZ_DF_rev.shape

# Label Encoding - to label all categorical variable value with numeric value
#Label will get assigned in Ascending alphabetical of variable value

colname1=['term','sub_grade','emp_length','home_ownership','verification_status',
          'purpose','zip_code','earliest_cr_line','last_pymnt_d',
          'next_pymnt_d','last_credit_pull_d','application_type']

XYZ_DF_rev.head()

from sklearn import preprocessing

le={}

for x in colname1:
     le[x]=preprocessing.LabelEncoder()

for x in colname1:
     XYZ_DF_rev[x]=le[x].fit_transform(XYZ_DF_rev[x])
XYZ_DF_rev.head()

#Train and Test split
#____________________________________________________________________________________________________________
# issue_d is object datatype to make use for split converting issue_d in Date
XYZ_DF_rev.issue_d = pd.to_datetime(XYZ_DF_rev.issue_d)   #%y-%m-%d
col_name = 'issue_d'
print (XYZ_DF_rev[col_name].dtype)



#split data in train and test

split_date = "2015-05-01"

XYZ_training = XYZ_DF_rev.loc[XYZ_DF_rev['issue_d'] <= split_date]
XYZ_training=XYZ_training.drop(['issue_d'],axis=1)
#XYZ_training.head()
XYZ_training.shape    #(598978, 40)

XYZ_test = XYZ_DF_rev.loc[XYZ_DF_rev['issue_d'] > split_date]
XYZ_test=XYZ_test.drop(['issue_d'],axis=1)
#XYZ_test.head()
XYZ_test.shape  #(256991, 40)



#selecting X and Y

X_train=XYZ_training.values[:,:-1]
Y_train=XYZ_training.values[:,-1]
Y_train=Y_train.astype(int)
print(Y_train)

X_test=XYZ_test.values[:,:-1]
Y_test=XYZ_test.values[:,-1]
Y_test=Y_test.astype(int)
print(Y_test)

#all reg module includes in sklearn.linear_model
from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
colname=XYZ_DF_rev.columns[:]
#fitting training data to the model
classifier.fit(X_train,Y_train)
#predicting on Test data
Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


#predicting using the Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier()
model_DecisionTree.fit(X_train,Y_train)

#fit the model on the data and predict the values
Y_pred=model_DecisionTree.predict(X_test)




#checking result
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)


from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier()
#model_GradientBoosting=DecisionTreeClassifier()

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)

#checking result
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)



#predicting using the AdaBoost_Classifier
from sklearn.ensemble import AdaBoostClassifier

model_AdaBoost=(AdaBoostClassifier(base_estimator=LogisticRegression(),n_estimators=10))

#base_estimator= specify algo
#n_estimators= by default 50
#fit the model on the data and predict the values
model_AdaBoost.fit(X_train,Y_train)

Y_pred=model_AdaBoost.predict(X_test)

#checking result
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)

y_pred_df=Y_pred[:]
#y_pred_df.columns=["Y Predicted value"]
y_pred_df

