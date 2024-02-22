#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Concrete compressive strength prediction in civil engineering

# In[2]:

# Important libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import itertools as it
from sklearn.linear_model import LinearRegression  # Linear regression
from sklearn.metrics import mean_squared_error, r2_score        # Compute mean square error, r2 score
from sklearn.model_selection import train_test_split   # Splitting dataset into training and test data
from sklearn.linear_model import Lasso              # Lasso Regression
from sklearn.neighbors import KNeighborsRegressor   # KNN Neighbor
from sklearn.svm import SVR                         # SVM
from sklearn.neural_network import MLPRegressor     # MLP
from sklearn import metrics
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

# In[4]:

# Loading of dataset
df = pd.read_csv('concrete_data.csv', sep=',')  # Create a dataframe
df.head(5)   # Reading of first 5 rows


values = df.columns.to_list()
# In[5]:






# Data Structuring
print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])
print(df.info())

# In[6]:

# Missing Values
print('Number of missing values:', df.isnull().sum())
'The dataset contains no missing values'

# In[7]:

# Data visualization 
# 1 Correlation Matrix
sns.heatmap(df.corr(), annot=True, linewidth=2)
plt.title("Correlation between variables")
plt.show()

# 2 Pair plot
sns.pairplot(df, markers="h")
plt.show()

# 3 Distribution plot
sns.distplot(df['concrete_compressive_strength'], bins=10, color='b')
plt.ylabel("Frequency")
plt.title('Distribution of concrete strength')

# In[8]:

# Distribution of components of concrete
cols = [i for i in df.columns if 'compressive_strength' not in i]  # Ensure this logic correctly filters out unwanted columns
length = len(cols)
cs = ["b", "r", "g", "c", "m", "k", "lime", "c"]
fig = plt.figure(figsize=(13, 25))

for i, j, k in zip(cols, range(length), cs):  # Use zip to safely iterate without exceeding the grid size
    plt.subplot(4, 2, j+1)
    ax = sns.histplot(data=df, x=i, color=k, kde=True)  # Updated to use histplot
    ax.set_facecolor("w")
    plt.axvline(df[i].mean(), linestyle="dashed", label="mean", color="k")
    plt.legend(loc="best")
    plt.title(i, color="navy")
    plt.xlabel("")


# In[9]:

# Scatterplot between components
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111)
plt.scatter(df["water"], df["cement"],
            c=df["concrete_compressive_strength"], s=df["concrete_compressive_strength"] * 3,
            linewidth=1, edgecolor="k", cmap="viridis")
ax.set_facecolor("w")
ax.set_xlabel("water")
ax.set_ylabel("cement")
lab = plt.colorbar()
lab.set_label("concrete_compressive_strength")
plt.title("cement vs water")
plt.show()

# In[10]:

# Data Splitting
# The dataset is divided into a 70 to 30 splitting between training data and test data
train, test = train_test_split(df, test_size=.3, random_state=0)
train_X = train[[x for x in train.columns if x not in ["concrete_compressive_strength"] + ["age_months"]]]
train_Y = train["concrete_compressive_strength"]
test_X = test[[x for x in test.columns if x not in ["concrete_compressive_strength"] + ["age_months"]]]
test_Y = test["concrete_compressive_strength"]

knn = KNeighborsRegressor()
model3 = knn.fit(train_X, train_Y)
predictions3 = knn.predict(test_X)
m13 = model3.score(test_X, test_Y)
RMSE13 = np.sqrt(metrics.mean_squared_error(test_Y, predictions3))
print('Accuracy of KNN model is', model3.score(test_X, test_Y))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, predictions3))
print('Mean Squared Error:', metrics.mean_squared_error(test_Y, predictions3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_Y, predictions3)))

# In[183]:

dat = pd.DataFrame({'Actual': test_Y, 'Predicted': predictions3})
dat1 = dat.head(25)  # just a sample which shows top 25 columns
dat1.plot(kind='bar', figsize=(7, 7))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# collect inputs and make a dictionary
input_dict = {}
for i in values[:len(values) - 1]:
    # print(f"Enter the value of {i}: ")
    input_dict[i] = float(input(f"Enter the value of {i}: "))


def predict_compressive_strength(model, **kwargs):
    """
    This function predicts the compressive strength of concrete based on the input values
    :param model: The trained model
    :param scaler: The trained scaler
    :param kwargs: The input values
    :return: The compressive strength of the concrete
    """
    # Create a dataframe from the input dictionary
    input_df = pd.DataFrame(kwargs, index=[0])
    # Predict the compressive strength
    prediction = model.predict(input_df)
    return prediction[0]


predicted_strength = predict_compressive_strength(knn, **input_dict)
print(f"The predicted compressive strength of the concrete is: {predicted_strength}")