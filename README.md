## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

<img width="461" height="448" alt="image" src="https://github.com/user-attachments/assets/8ff57776-1fbd-4f7c-a5b1-47374cab870c" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]]

```

<img width="194" height="236" alt="image" src="https://github.com/user-attachments/assets/98a9e493-9734-423a-a999-2745d2de5df7" />

```

df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="397" height="446" alt="image" src="https://github.com/user-attachments/assets/986233db-250e-4d42-8505-d5420f2298bc" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="404" height="454" alt="image" src="https://github.com/user-attachments/assets/7a1b85e7-2658-480a-b16b-b900b97b23a2" />


```

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

```
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="526" height="446" alt="image" src="https://github.com/user-attachments/assets/ce05b265-e737-4418-b95b-ba65c3727477" />

```

pd.get_dummies(df2,columns=["nom_0"])
```

<img width="782" height="440" alt="image" src="https://github.com/user-attachments/assets/3a50f0ea-62c2-4dec-a7cf-863137e075ba" />

```

pip install --upgrade category_encoders
```

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

<img width="585" height="451" alt="image" src="https://github.com/user-attachments/assets/1363268c-56ad-4353-b134-a5ba511c57d1" />

```

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

<img width="584" height="452" alt="image" src="https://github.com/user-attachments/assets/8585af53-5a16-4866-942c-cb9337eb9b54" />

```

dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="832" height="445" alt="image" src="https://github.com/user-attachments/assets/df83b3a5-3229-415b-a5ce-89379c6d880e" />


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="681" height="438" alt="image" src="https://github.com/user-attachments/assets/ac44229f-8fe9-4363-82f7-7ddb401cebe2" />

```

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="915" height="541" alt="image" src="https://github.com/user-attachments/assets/83d2dae7-af76-410b-af5c-a84cf76f72c4" />

```

df.skew()
```

<img width="360" height="216" alt="image" src="https://github.com/user-attachments/assets/d92b1dd0-4a11-41ce-a759-54db353652e6" />

```

np.log(df["Highly Positive Skew"])
```

<img width="329" height="568" alt="image" src="https://github.com/user-attachments/assets/b9d9709a-9260-499a-a00a-c20051802e6e" />

```

np.reciprocal(df["Moderate Positive Skew"])
```

<img width="387" height="578" alt="image" src="https://github.com/user-attachments/assets/6d8468ae-0d61-4f52-844b-c0ec13e0777f" />

```

np.sqrt(df["Highly Positive Skew"])
```

<img width="335" height="561" alt="image" src="https://github.com/user-attachments/assets/e90641cc-62f5-490f-a265-74ba7cc08cf1" />

```

np.square(df["Highly Positive Skew"])
```

<img width="337" height="557" alt="image" src="https://github.com/user-attachments/assets/dda0eb64-bc54-49c8-b1ae-f17ac13d6b7a" />

```

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="928" height="532" alt="image" src="https://github.com/user-attachments/assets/bc344c0c-7843-4f33-853b-5a99428bc4b4" />

```

df.skew()
```

<img width="409" height="273" alt="image" src="https://github.com/user-attachments/assets/f3a3d202-859b-4819-a392-bfd34034a23d" />

```

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="433" height="331" alt="image" src="https://github.com/user-attachments/assets/5a800704-e082-4c4c-b75b-d42300e36f6e" />

```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="907" height="559" alt="image" src="https://github.com/user-attachments/assets/6e8d0b63-d7df-4a1a-b68d-179669e8fd4b" />

```

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="748" height="553" alt="image" src="https://github.com/user-attachments/assets/1d53e12c-0128-49f6-a5e0-22efca4a4bd5" />

```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="739" height="552" alt="image" src="https://github.com/user-attachments/assets/965ed0aa-d115-477a-9a5e-29fe8284ebb2" />


```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="718" height="546" alt="image" src="https://github.com/user-attachments/assets/fc05c4cd-2df6-45bd-a577-6b0b3df74514" />

```

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="746" height="562" alt="image" src="https://github.com/user-attachments/assets/451e795d-473f-4a88-97af-1eec74250249" />

```

dt=pd.read_csv("titanic_dataset.csv")
dt
```

<img width="1448" height="525" alt="image" src="https://github.com/user-attachments/assets/f674f2d0-ba0d-40fb-b226-2a458019d714" />

```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```

<img width="722" height="552" alt="image" src="https://github.com/user-attachments/assets/30f64f59-ae4e-4353-ba30-e0b3cb497989" />

```

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="718" height="537" alt="image" src="https://github.com/user-attachments/assets/6d6c88bb-fe96-4bd4-be8e-52d0c1f6e036" />





# RESULT

Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully

       
