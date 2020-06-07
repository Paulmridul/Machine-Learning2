# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # MLR
# 
# %% [markdown]
# ## Importing Libraries

# %%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# %% [markdown]
# ## Importing DataSet

# %%
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# %% [markdown]
# ### Some Dataset details check

# %%
print(dataset.shape)
dataset.head()


# %%
print(dataset.describe())


# %%
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()


# %%
dataset.hist()
plt.show()

# %% [markdown]
# ## Encoding Catagorical Data

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x = np.array(ct.fit_transform(x))


# %%
print(x)

# %% [markdown]
# ## Spliting Dataset into Training Set and Test Set

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# %% [markdown]
# # Training the Multiple linear regression model on the Training set

# %%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# %% [markdown]
# # Predicting the Test set results

# %%
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# ## Find R2 Value for the model

# %%
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
print(score)


# %%


