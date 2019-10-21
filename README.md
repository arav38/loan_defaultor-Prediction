import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the data
loan = pd.read_excel(r"C:\akshay\Data Science Final Project\XYZCorp_LendingData .xlsx")

loan.head(5)

#making the copy of orignal file
loan_copy = pd.DataFrame.copy(loan)

# to know about the no of column and rows
loan_copy.shape

#To get only half dataset
half_count = len(loan_copy) / 2 
 ## Drop column's with more than 50% missing values
loan_copy= loan_copy.dropna(thresh=half_count,axis=1)
loan_copy.head()

loan_copy.shape

#loan_copy.isnull().sum()
loan_copy.dtypes

loan_copy.isnull().sum()

#handling the missing values

for x in loan_copy[:]:
    if loan_copy[x].dtype=='object':
            loan_copy[x].fillna(loan_copy[x].mode()[0],inplace=True)
    elif loan_copy[x].dtype=='int64' or loan_copy[x].dtype=='float64':
            loan_copy[x].fillna(loan_copy[x].mean(),inplace=True)
            
loan_copy.isnull().sum()


loan_copy = loan_copy.drop(["last_credit_pull_d","next_pymnt_d","next_pymnt_d","last_pymnt_d"], axis = 1)

loan_copy.isnull().sum()

loan_copy.dtypes

#Converting all the float/object data type in interger format

colname = ["id","member_id",
"loan_amnt",
"funded_amnt",
"funded_amnt_inv",
"term",
"int_rate",
"installment",
"annual_inc",
"issue_d",
"dti",
"delinq_2yrs",
"earliest_cr_line",
"inq_last_6mths",
"open_acc",
"pub_rec",
"revol_bal",
"revol_util",
"total_acc",
"out_prncp",
"out_prncp_inv",
"total_pymnt",
"total_pymnt_inv",
"total_rec_prncp",
"total_rec_int",
"total_rec_late_fee",
"recoveries",
"collection_recovery_fee",
"last_pymnt_amnt",
"collections_12_mths_ex_med",
"policy_code",
"acc_now_delinq",
"tot_coll_amt",
"tot_cur_bal",
"total_rev_hi_lim",
"default_ind"
]

## Preproccessig of data using sklearn library/Converting cat data to numeric

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for x in colname:

    loan_copy[x] =le.fit_transform(loan_copy[x])
    
loan_copy.head()

loan_copy.dtypes

# splitting of data in train and test

# setting the index as date
loan_copy['issue_d'] = pd.to_datetime(loan_copy.issue_d,format='%Y-%m-%d')
loan_copy.index = loan_copy['issue_d']

#data = loan_copy.sort_index(ascending=True, axis=0)
# create train test partition
train = loan_copy['2007-06-01':'2015-05-01']
test  = loan_copy['2015-06-01':]
print('Train Dataset:',train.shape)
print('Test Dataset:',test.shape)


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(train)

train =scaler.transform(train)
print(train)
