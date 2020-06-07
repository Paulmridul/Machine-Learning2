# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # SVR
# %% [markdown]
# ## Importing Libraries

# %%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# %% [markdown]
# ## Importing the dataset

# %%
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2:].values


# %%
print(x)


# %%
print(y)

# %% [markdown]
# ## Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_fs = sc_x.fit_transform(x)
sc_y = StandardScaler()
y_fs = sc_y.fit_transform(y)


# %%
print(x_fs)


# %%
print(y_fs)

# %% [markdown]
# ## Training the SVR model on the whole dataset

# %%
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x_fs,y_fs)

# %% [markdown]
# ## Predicting a new result

# %%
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

# %% [markdown]
# ## Visualising the SVR results

# %%
plt.scatter(x,y,c='red')
plt.plot(x, sc_y.inverse_transform(regressor.predict(x_fs)),color='blue')
plt.title('Support Vector Regression Result')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# %% [markdown]
# ## Visualising the SVR results (for higher resolution and smoother curve)

# %%
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# %%


