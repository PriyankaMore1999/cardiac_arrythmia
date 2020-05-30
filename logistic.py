#logistic
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

data = pd.read_excel('data.xlsx',sep="\t",header=None,decimal= ".")
X = data.iloc[:, 0:277].values
y = data.iloc[:, 278].values

a=np.asarray(X)
b=np.asarray(y)


data2=data.loc[(data[278]!=1)]   
X1 = data2.iloc[:, 0:277].values
y1=data2.iloc[:, 278].values

a1=np.asarray(X1)
b1=np.asarray(y1)

b[b >= 2] = 2
print("using Logistic Regression")

def cardiac(m,n):
	imputer = SimpleImputer(missing_values = np.nan, strategy = 'median',verbose=0)
	imputer = imputer.fit(m)
	m = imputer.transform(m)


	X_train, X_test, y_train, y_test = train_test_split(m, n, test_size = 0.25, random_state = 100)

	s = StandardScaler()										#feature scaling
	X_train = s.fit_transform(X_train)
	X_test = s.transform(X_test) 

	model2 = LogisticRegression(solver='liblinear')				#logistic regression model
	model2.fit(X_train, y_train)
	
	b2 = cross_validate(model2, X_train, y_train, cv=5)			#cross validate
	print(sorted(b2.keys()))
	print(b2['test_score'])


	def error_rate(model2,y_test,X_test):      # Function to calculate error_rate
		y_pred = model2.predict(X_test)
		error = 1.0-metrics.accuracy_score(y_test,y_pred)
		return error

	print("Error_rate:",(error_rate(model2,y_test,X_test))*100,' %')    # Function to calculate Error_rate
	print("Accuracy:",(1-error_rate(model2,y_test,X_test))*100,' %')    # Function to calculate Accuracy of model
	print('Confusion Matrix: \n',confusion_matrix(y_test,model2.predict(X_test)))# to print the Confusion matrix
	print('\n')

	FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
	FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
	TP = np.diag(confusion_matrix)
	TN = confusion_matrix.values.sum() - (FP + FN + TP)
	print("True negative % ",TN/(TN+FP) )

cardiac(a,b)

cardiac(a1,b1)