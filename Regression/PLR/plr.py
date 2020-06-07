# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Polynomial linear Regression
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
y = dataset.iloc[:, -1].values

# %% [markdown]
# ## Training the Linear Regression model on the whole dataset

# %%
from sklearn.linear_model import  LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# %% [markdown]
# ## Training the Polynomial Regression model on the whole dataset

# %%
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# %% [markdown]
# ## Visualising the Linear Regression results

# %%
plt.scatter(x,y,c='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear Regression Result')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# %% [markdown]
# ## Visualising the Polynomial Regression results

# %%
plt.scatter(x,y,c='red')
plt.plot(x, lin_reg_2.predict(x_poly),color='blue')
plt.title('Polynomial Regression Result')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# %% [markdown]
# ## Visualising the Polynomial Regression results (for higher resolution and smoother curve)

# %%
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# %% [markdown]
# ## Predicting a new result with Linear Regression

# %%
lin_reg.predict([[6.5]])

# %% [markdown]
# ## Predicting a new result with Polynomial Regression

# %%
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

# %% [markdown]
# ## Find R2 Value for the linear Model

# %%
from sklearn.metrics import r2_score
score = r2_score(y,lin_reg.predict(x))
print(score)

# %% [markdown]
# ## Find R2 Value for the polynomial Model

# %%
score = r2_score(y,lin_reg_2.predict(poly_reg.fit_transform(x)))
print(score)


# %%


