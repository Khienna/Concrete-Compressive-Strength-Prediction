# Download dependencies.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import itertools as it
from sklearn.metrics import mean_squared_error, r2_score 
# Compute mean square error, r2 score
from sklearn.model_selection import train_test_split   
# Splitting dataset into training and test data
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR                         # SVM
from sklearn import metrics
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

#Loading of dataset
df = pd.read_csv('concrete_data.csv', sep=',')  
# Create a dataframe
df.head(5)   
# Reading of first 5 rows


values = df.columns.to_list()

# Data Structuring
print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])
print(df.info())


# Check for Missing Values in the data
print('Number of missing values:', df.isnull().sum())
# 'The dataset contains no missing values.

train, test = train_test_split(df, test_size=.3, random_state=0)
train_X = train[[x for x in train.columns if x not in ["concrete_compressive_strength"] + ["age_months"]]]
train_Y = train["concrete_compressive_strength"]
test_X = test[[x for x in test.columns if x not in ["concrete_compressive_strength"] + ["age_months"]]]
test_Y = test["concrete_compressive_strength"]

svm = SVR(kernel='linear')
model4 = svm.fit(train_X, train_Y)
predictions4 = svm.predict(test_X)
m4 = model4.score(test_X, test_Y)
RMSE4 = np.sqrt(metrics.mean_squared_error(test_Y, predictions4))
print('Accuracy of model is', model4.score(test_X, test_Y))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, predictions4))
print('Mean Squared Error:', metrics.mean_squared_error(test_Y, predictions4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_Y, predictions4)))

# In[185]:

dat = pd.DataFrame({'Actual': test_Y, 'Predicted': predictions4})
dat1 = dat.head(25)  # just a sample which shows top 25 columns
dat1.plot(kind='bar', figsize=(7, 7))
plt.grid(which='major', linestyle='-', linewidth='0.5',

 color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


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


predicted_strength = predict_compressive_strength(svm, **input_dict)
print(f"The predicted compressive strength of the concrete is: {predicted_strength}")
