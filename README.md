# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NIRANJAN S
RegisterNumber:  212224040221
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![decision tree classifier model](sam.png)
## data.head()
![318640788-a5281883-eed0-439e-b4fa-922bbdfbed98](https://github.com/user-attachments/assets/119d641c-8464-4519-84e5-0100c7ec1e8c)
## data.info()

![318640974-7c989f04-c2db-4949-a088-ad348bbec7c7](https://github.com/user-attachments/assets/d55d09a1-0543-404d-8066-d2ddedb9e85d)
## data.isnull().sum()
![318641268-3d23a2f9-d8bf-4ca2-b00c-3e581ca80770](https://github.com/user-attachments/assets/2855f5cf-2e15-4de6-abca-8c53ac97700e)
## data value count
![318641864-10ee6c65-157b-4f16-95bc-0dc9664953a9](https://github.com/user-attachments/assets/7ac7409e-3973-4109-9870-5e34be8a923b)
## data.head() for salary
![318642349-6492ed67-18af-46c1-8141-d168878dd59d](https://github.com/user-attachments/assets/711b118e-4b1d-4239-9355-f445c2018163)
## x.head()
![318642492-b3b08f53-9a93-4cdc-8403-6c601bae877e](https://github.com/user-attachments/assets/f0f5729a-898e-4fea-ba25-b8f734f9c7df)
## accuracy value
![318642622-7026b909-59d8-4316-afaf-22270aa8dd90](https://github.com/user-attachments/assets/5e6c12f8-7c7d-4d3e-a13c-06cf55b3a398)
## data prediction
![318642746-5f8fee59-9658-437e-8a98-8405059f69c6](https://github.com/user-attachments/assets/47df8cad-e4a1-4270-b4bb-a390e7353e58)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
