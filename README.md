import pandas as pd
import numpy as np

loan = pd.read_excel(r"C:\akshay\Data Science Final Project\XYZCorp_LendingData1.xlsx")

loan.head()
    #making the copy of orignal file
loan_copy = pd.DataFrame.copy(loan)
loan_copy.shape
    
    #To get only half dataset
half_count = len(loan_copy) / 2 
 ## Drop column's with more than 50% missing values
loan_copy= loan_copy.dropna(thresh=half_count,axis=1)
loan_copy.head()

loan_copy.shape

loan_copy.isnull().sum()

loan_copy = loan_copy.drop(["last_credit_pull_d","next_pymnt_d","last_pymnt_d","id","member_id",'funded_amnt','funded_amnt_inv','emp_title'], axis = 1)

for x in loan_copy[:]:
    if loan_copy[x].dtype=='object':
            loan_copy[x].fillna(loan_copy[x].mode()[0],inplace=True)
    elif loan_copy[x].dtype=='int64' or loan_copy[x].dtype=='float64':
            loan_copy[x].fillna(loan_copy[x].mean(),inplace=True)
            
loan_copy.isnull().sum()
#loan_copy.emp_title.unique()
    
loan_copy['application_type'] = loan_copy['application_type'].replace({'INDIVIDUAL':1,'JOINT':2})
loan_copy['grade'] = loan_copy['grade'].replace({'A':7,'B':6,'C':5,'D':4,'E':3,'F':2,'G':1})
loan_copy["home_ownership"] = loan_copy["home_ownership"].replace({"MORTGAGE":6,"RENT":5,"OWN":4,"OTHER":3,"NONE":2,"ANY":1})
loan_copy["emp_length"] = loan_copy["emp_length"].replace({'years':'','year':'',' ':'','<':'','\+':'','n/a':'0'}, regex = True)
loan_copy["emp_length"] = loan_copy["emp_length"].apply(lambda x:int(x))
print("Current shape of dataset :",loan_copy.shape)
loan_copy.head()
    
    
#converting ['issue_d'] into the datetime format 
loan_copy.issue_d=pd.to_datetime(loan_copy.issue_d)
col_name = 'issue_d'
print(loan_copy[col_name].dtype)
    
    
    

#pre-processing
colname=[]
for x in loan_copy.columns[:]:
    if loan_copy[x].dtype=='object':
        colname.append(x)
colname

## Preproccessig of data using sklearn library/Converting cat data to numeric

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for x in colname:

    loan_copy[x] =le.fit_transform(loan_copy[x])
    
loan_copy.head()

#split the data

split_date = "2015-06-01"
xyz_train=loan_copy.loc[loan_copy['issue_d'] < split_date]
xyz_train=xyz_train.drop(['issue_d'],axis=1)
xyz_train.shape   #(598978, 39)

xyz_test=loan_copy.loc[loan_copy['issue_d'] >= split_date]
xyz_test=xyz_test.drop(['issue_d'],axis=1)
xyz_test.shape #(256991, 39)



X_train=xyz_train.values[:,:-1]
Y_train=xyz_train.values[:,-1]
Y_train=Y_train.astype(int)
print(Y_train)


X_test=xyz_test.values[:,:-1]
Y_test=xyz_test.values[:,-1]
Y_test=Y_test.astype(int)
print(Y_test)


#SCALING

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

scaler.fit(X_train)

X_train=scaler.transform(X_train)

X_test=scaler.transform(X_test)

#LOGISTIC REGRESSION 

from sklearn.linear_model import LogisticRegression
#create a model
classifier = LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

print(list(zip(Y_test,Y_pred)))
#this gives the slope of each varibale 
print(classifier.coef_)
#this gives a intercept 
print(classifier.intercept_)   


from sklearn.metrics import confusion_matrix, accuracy_score , classification_report

cfm = confusion_matrix(Y_test,Y_pred)

print(cfm)
