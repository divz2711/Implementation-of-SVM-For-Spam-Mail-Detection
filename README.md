# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Extract relevant statistical and domain-specific features.
2. Include basic statistics (mean, median, std. deviation).
3. Handle missing values (impute with mean, median, etc.).
4. Normalize/standardize features for consistent scale.
5. Optionally, perform dimensionality reduction (e.g., PCA).
6. Split data into training and testing sets.
7. Output: Feature matrix for machine learning.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Divya S
RegisterNumber:  212221040042
*/

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(1000000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy


```

## Output:
## Result output
![image](https://github.com/divz2711/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121245222/a86576dc-d29f-4e97-bc14-a51a2e8d9b22)

## data.head()
![image](https://github.com/divz2711/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121245222/fc2ac872-2ad5-410b-8893-7de07c8d0c8e)

## data.info()
![image](https://github.com/divz2711/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121245222/d5599d9f-30d7-44c4-a1ad-c7d2e0e40062)

## data.isnull().sum()
![image](https://github.com/divz2711/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121245222/5056789b-babb-4f44-be19-55976463d6d1)

## Y_prediction value
![image](https://github.com/divz2711/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121245222/9cd52749-0035-4cf7-9b86-130d751682ae)

## Accuracy value
![image](https://github.com/divz2711/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121245222/8f4896f2-c3ff-4cad-b09e-fb4edb7cc751)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
