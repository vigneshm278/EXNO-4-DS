# EXNO:4-DS
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



# RESULT:
       # INCLUDE YOUR RESULT HERE
