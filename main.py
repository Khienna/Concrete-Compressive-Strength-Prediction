
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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import statsmodels.api as sm

# In[4]:

# Loading of dataset
df = pd.read_csv('concrete_data.csv', sep=',')  # Create a dataframe
df.head(5)   # Reading of first 5 rows

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

# In[206]:

# Model 1 = Multiple linear regression
# fit a model
lm = LinearRegression()
model = lm.fit(train_X, train_Y)
predictions = lm.predict(test_X)
m1 = model.score(test_X, test_Y)
RMSE1 = np.sqrt(metrics.mean_squared_error(test_Y, predictions))
print('Accuracy of model MLR is', model.score(test_X, test_Y))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(test_Y, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_Y, predictions)))

# In[177]:

# Features Importance
coef = pd.DataFrame(lm.coef_.ravel

())
coef['feat'] = train_X.columns
c = coef['feat'].rows=['C', 'Bfs', 'Fa', 'W', 'Sp', 'Ca', 'Fa', 'a']
num_colors = 8
colors = cm.rainbow(np.linspace(0, 1, num_colors))
plt.bar(c, coef[0], color=colors)
plt.show()
'where C: Cement, Bfs: Blast_furnace_slag, Fa: Fly_ash, W: water, Sp: Superplasticizer, Ca: Coarse_aggregate, Fa: Fine_aggregate, a: Age'

# In[178]:

# Plot of true value vs. predicted values
dat = pd.DataFrame({'Actual': test_Y, 'Predicted': predictions})
dat1 = dat.head(25)  # just a sample which shows top 25 columns
dat1.plot(kind='bar', figsize=(7, 7))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# In[205]:

# Model 2: LASSO Model
las = Lasso(alpha=0.1)
model2 = las.fit(train_X, train_Y)
predictions2 = las.predict(test_X)
m12 = model2.score(test_X, test_Y)
RMSE12 = np.sqrt(metrics.mean_squared_error(test_Y, predictions2))
print('Accuracy of model LASSO is', model2.score(test_X, test_Y))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, predictions2))
print('Mean Squared Error:', metrics.mean_squared_error(test_Y, predictions2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_Y, predictions2)))

# In[180]:

# Feature selection
coef1 = pd.DataFrame(las.coef_.ravel())
coef1['feat'] = train_X.columns
c1 = coef['feat'].rows=['C', 'Bfs', 'Fa', 'W', 'Sp', 'Ca', 'Fa', 'a']
num_colors = 8
colors = cm.rainbow(np.linspace(0, 1, num_colors))
plt.bar(c1, coef[0], color=colors)
plt.show()

# In[181]:

# Plot of true value vs. predicted values
dat = pd.DataFrame({'Actual': test_Y, 'Predicted': predictions2})
dat1 = dat.head(25)  # just a sample which shows top 25 columns
dat1.plot(kind='bar', figsize=(7, 7))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# In[204]:

# Model 3: KNN Neighbor
knn = KNeighborsRegressor()
model3 = knn.fit(train_X, train_Y)
predictions3 = knn.predict(test_X)
m13 = model3.score(test_X, test_Y)
RMSE13 = np.sqrt(metrics.mean_squared_error(test_Y, predictions3))
print('Accuracy of model KNN is', model3.score(test_X, test_Y))
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

# In[ ]:

#Model 4: SVM.
svm = SVR(kernel='linear')
model4 = svm.fit(train_X, train_Y)
predictions4 = svm.predict(test_X)
m4 = model4.score(test_X, test_Y)
RMSE4 = np.sqrt(metrics.mean_squared_error(test_Y, predictions4))
print('Accuracy of model SVR is', model4.score(test_X, test_Y))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, predictions4))
print('Mean Squared Error:', metrics.mean_squared_error(test_Y, predictions4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_Y, predictions4)))

# In[185]:





# Add Model 5: Multilayer Perceptron (MLP)
# In[250]:

print(df.columns)
X = df.drop('concrete_compressive_strength', axis=1)
Y = df['concrete-compressive-strength']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train_scaled, train_Y)
predictions_mlp = mlp.predict(test_X)

# Model Evaluation for MLP
m5 = mlp.score(test_X, test_Y)
RMSE5 = np.sqrt(mean_squared_error(test_Y, predictions_mlp))
print('Accuracy of MLP model is', m5)
print('Mean Absolute Error:', metrics.mean_absolute_error(test_Y, predictions_mlp))
print('Mean Squared Error:', metrics.mean_squared_error(test_Y, predictions_mlp))
print('Root Mean Squared Error:', RMSE5)


# In[223]:

def predict_compressive_strength(model, scaler, **kwargs):
    input_df = pd.DataFrame([kwargs])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]


input = {
    'cement': 540,
    'blast_furnace_slag': 0,
    'fly_ash': 0,
    'water': 162,
    'superplasticizer': 2.5,
    'coarse_aggregate': 1040,
    'fine_aggregate': 676,
    'age': 28
}

predicted_strength = predict_compressive_strength(mlp, scaler=scaler, **input)
print(f'Predicted compressive strength: {predicted_strength} MPa')
# Model comparison
# heights = [m1, m12, m13, m4, m5]
# bars = ('Linear', 'Poly2', 'Poly3', 'Lasso', 'KNN', 'SVM', 'MLP')
# y_pos = np.arange(len(bars))
# plt.bar(y_pos, heights)
# plt.xticks(y_pos, bars)
# plt.title('Model Comparison')
# plt.xlabel('Models')
# plt.ylabel('Accuracy')
# plt.show()


# In[ ]:


# Model comparison
# heights = [RMSE1, RMSE2, RMSE3, RMSE12, RMSE13, RMSE4, RMSE5]
# bars = ('Linear', 'Poly2', 'Poly3', 'Lasso', 'KNN', 'SVM', 'MLP')
# y_pos = np.arange(len(bars))
# plt.bar(y_pos, heights)
# plt.xticks(y_pos, bars)
# plt.title('Model Comparison')
# plt.xlabel('Models')
# plt.ylabel('RMSE')
# plt.show()

