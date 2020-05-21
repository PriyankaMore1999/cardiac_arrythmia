import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

data = pd.read_excel('data.xlsx',sep="\t",header=None,decimal= ".")
data1=[]
classifier=['SVM','Logistic Regression']

X = data.iloc[:, 0:278].values
y = data.iloc[:, 278].values
a=np.asarray(X)
b=np.asarray(y)

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(a)
a = imputer.transform(a)


X_train, X_test, y_train, y_test = train_test_split(a, b, test_size = 0.25, random_state = 100)

model = SVC()
model.fit(X_train, y_train)
data1.append(model)
'''
model1 = BernoulliNB()
model1.fit(X_train, y_train)
data1.append(model1)
'''
model2 = LogisticRegression(verbose=1,n_jobs=-1,solver='saga',penalty='l2',max_iter=1000)
model2.fit(X_train, y_train)
data1.append(model2)

def error_rate(model,y_test,X_test):      # Function to calculate error_rate
    y_pred = model.predict(X_test)
    error = 1.0-metrics.accuracy_score(y_test,y_pred)
    return error


for (i,j) in zip(data1,classifier): 
    print('Using ',j,' classifier:\n')
    print("Error_rate:",(error_rate(i,y_test,X_test))*100,' %')    # Function to calculate Error_rate
    print("Accuracy:",(1-error_rate(i,y_test,X_test))*100,' %')    # Function to calculate Accuracy of model
    print('Confusion Matrix: \n',confusion_matrix(y_test,i.predict(X_test)))# to print the Confusion matrix
	
    print('\n')
