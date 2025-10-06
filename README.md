# EXNO:4 - Feature Scaling and Selection
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
Name: Vignesh M
Ref no: 212223240176

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

v=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
v
```
<img width="893" height="376" alt="image" src="https://github.com/user-attachments/assets/a49b776c-6353-4e73-924d-728a3f01c251" />

```
v.isnull().sum()
```
<img width="472" height="317" alt="image" src="https://github.com/user-attachments/assets/3168a23d-464a-4255-9cbb-ede0770e4838" />

```
missing=v[v.isnull().any(axis=1)]
missing
```
<img width="801" height="301" alt="image" src="https://github.com/user-attachments/assets/508e82b9-bc33-42b0-b6a1-f938fca17e0f" />

```
v2=v.dropna(axis=0)
v2
```
<img width="844" height="411" alt="image" src="https://github.com/user-attachments/assets/2718c5fb-48ec-4f74-9f14-fb7a723d6693" />

```
sal=v["SalStat"]

v2["SalStat"]=v["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(v2['SalStat'])
```
<img width="811" height="208" alt="image" src="https://github.com/user-attachments/assets/0a451316-7823-4130-995f-62f696c8aa5f" />

```
sal2=v2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="563" height="518" alt="image" src="https://github.com/user-attachments/assets/20d940d8-6bd6-4c6b-a3e4-bbd7e2f773f8" />

```
new_data=pd.get_dummies(v2, drop_first=True)
new_data
```
<img width="870" height="298" alt="image" src="https://github.com/user-attachments/assets/17df9162-82e1-4149-9fe7-03a2c2db322f" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="865" height="33" alt="image" src="https://github.com/user-attachments/assets/39a7bfef-c811-4c3b-aa17-4f891d8d74bf" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="884" height="28" alt="image" src="https://github.com/user-attachments/assets/074e907d-ffc4-4400-898b-82d639658521" />

```
y=new_data['SalStat'].values
print(y)
```
<img width="145" height="26" alt="image" src="https://github.com/user-attachments/assets/d8fb6c9a-a75a-4034-9690-d632b27426d6" />

```
x=new_data[features].values
print(x)
```
<img width="349" height="88" alt="image" src="https://github.com/user-attachments/assets/825eb107-ece7-4be3-90e3-a89a60ad1dcd" />

```

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
<img width="249" height="52" alt="image" src="https://github.com/user-attachments/assets/834c7684-2fdb-48d9-9074-e2283bf19ef8" />


```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="205" height="31" alt="image" src="https://github.com/user-attachments/assets/7590a4bc-2a7a-4713-8f28-04b5155fbe6f" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="160" height="24" alt="image" src="https://github.com/user-attachments/assets/e2671b28-b2f4-42fc-a62b-35fdbded0d8a" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="226" height="21" alt="image" src="https://github.com/user-attachments/assets/e5a91a48-b51f-42eb-9167-f762f4f76753" />

```
data.shape
```

<img width="131" height="24" alt="image" src="https://github.com/user-attachments/assets/a43527ae-36e4-4c8e-a8c3-e2adf45fed38" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="720" height="59" alt="image" src="https://github.com/user-attachments/assets/47fe60f0-511e-4a7e-ae09-6bfe9621836d" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="479" height="131" alt="image" src="https://github.com/user-attachments/assets/04dbff82-6263-49ef-8398-692a29963f56" />


```
tips.time.unique()
```
<img width="294" height="34" alt="image" src="https://github.com/user-attachments/assets/bce4f72c-7e78-4d58-9ff5-56fac968e599" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="217" height="62" alt="image" src="https://github.com/user-attachments/assets/9d492ca0-d5ac-4bf6-9be6-67fa0f87e1d8" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")`

```
<img width="286" height="40" alt="image" src="https://github.com/user-attachments/assets/b91ea18c-c440-48f6-b7aa-31f65708b9da" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="307" height="36" alt="image" src="https://github.com/user-attachments/assets/e20708d0-6c4b-4d82-9fbf-95ab95a49e3a" />



# RESULT:

Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
