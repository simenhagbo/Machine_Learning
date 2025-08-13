# My first ML project
# Import pandas and the data file
import matplotlib
import pandas as pd
from pandas.core.common import random_state

url = 'https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv'
df = pd.read_csv(url)

# Data preperation
# Data seperation as X and y
y = df['logS'] # Data the model is going to predict
X = df.drop('logS', axis=1) # Data the model will use to learn

# Data splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print(df.shape)
print(X_train.shape)
print(X_test.shape)

# Model building
# Linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train) # Tell the model to learn the connection between learning and solution

# Applying model to make prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print("LR Train mse: ", lr_train_mse)
print("LR Train r2: ", lr_train_r2)
print("LR Test mse: ", lr_test_mse)
print("LR Test r2: ", lr_test_r2)

lr_results = pd.DataFrame([['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]], columns=['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'])



# Building a random forest model
# Training the model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=13, random_state=100)
rf.fit(X_train, y_train)
# Applying the model to make a prediction
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

# Evaluate the models performance
from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame([['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]], columns=['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'])

# Model comparison
df_models = pd.concat([lr_results, rf_results], axis=0)
print(df_models)

df_models = df_models.reset_index(drop=True)
print(df_models)

# Data visualization
import matplotlib.pyplot as plt
import numpy as np
# Visualizate Linear regression
plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt. xlabel('Experimental LogS')
plt.title('Linear Regression - Training')
plt.show()

# Visualizate random forest training
plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_rf_train_pred, alpha=0.3)

z = np.polyfit(y_train, y_rf_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#00BFC4')
plt.ylabel('Predict LogS(RF)')
plt.xlabel('Experimental LogS')
plt.title('Random Forest - Training')
plt.show()

# Visualize RF test data
plt.figure(figsize=(5,5))
plt.scatter(x=y_test, y=y_rf_test_pred, alpha=0.3)

z = np.polyfit(y_test, y_rf_test_pred, 1)
p = np.poly1d(z)

plt.plot(y_test, p(y_test), '#00BFC4')
plt.ylabel('Predicted LogS (RF - Test)')
plt.xlabel('Experimental LogS')
plt.title('Random Forest - Test')
plt.show()
