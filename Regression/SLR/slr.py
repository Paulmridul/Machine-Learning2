# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # SLR
# %% [markdown]
# ## Importing Libraries

# %%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# %% [markdown]
# ## Importing DataSet

# %%
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# %% [markdown]
# ## Spliting Dataset into Training Set and Test Set

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# %% [markdown]
# ## Training the Simple linear regression model on the Training set

# %%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# %% [markdown]
# ## Predicting the Test Set Result

# %%
y_pred = regressor.predict(x_test)

# %% [markdown]
# ## Visualising Training Set Result

# %%
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# %% [markdown]
# ## Visualising Test Set Result
# 

# %%
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# %% [markdown]
# ## Find R2 Value for the Model
# 

# %%
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
print(score)


# %%


