# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Data PreProcessing Tool
# %% [markdown]
# ## Importing Libraries

# %%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# %% [markdown]
# ## Importing the dataset

# %%
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[ :, :-1].values
y = dataset.iloc[ :, -1].values


# %%
print(x)


# %%
print(y)

# %% [markdown]
# ## Taking care of missing data

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])


# %%
print(x)

# %% [markdown]
# ## Encoding categorical data
# %% [markdown]
# ### Encoding the Independent Variable

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x))


# %%
print(x)

# %% [markdown]
# ### Encoding the Dependent Variable

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# %%
print(y)

# %% [markdown]
# ## Splitting the dataset into the Training set and Test set

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# %%
print(x_train)


# %%
print(x_test)


# %%
print(y_train)


# %%
print(y_test)

# %% [markdown]
# ## Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])


# %%
print(x_train)


# %%
print(x_test)


# %%


