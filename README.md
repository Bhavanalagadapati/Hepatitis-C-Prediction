# Hepatitis-C-Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing


data=pd.read_csv('/content/hepatitisc.zip')


data


from sklearn.preprocessing import LabelEncoder
target_column='Category'
label_encoder=LabelEncoder()
data[target_column]=label_encoder.fit_transform(data[target_column])


data.shape


data.info()


data.head()

data.tail()

data.describe


data["Age"].value_counts()

print('The highest unnamed:0 was of:',data['Unnamed: 0'].max())
print('The lowest unnamed:0 was of:',data['Unnamed: 0'].min())
print('The average unamed:0 in the data:',data['Unnamed: 0'].mean())

data.duplicated()

data.isnull().sum()

data=data.fillna(method='ffill')
data

data.isnull().sum()

#line plot
plt.plot(data['CREA'])
plt.xlabel("CREA")
plt.ylabel("Levels")
plt.title("Line Plot")
plt.show()

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Conditions for individuals with Hepatitis (Category 1, 2, or 3)
hepatitis_conditions = (data['Category'] == 1) | (data['Category'] == 2) | (data['Category'] == 3)
data_len_hepatitis = data[hepatitis_conditions]['ALP'].value_counts()

ax1.hist(data_len_hepatitis, color='red')
ax1.set_title('Having Hepatitis')

# Conditions for individuals without Hepatitis (Category 0)
no_hepatitis_conditions = (data['Category'] == 0)
data_len_no_hepatitis = data[no_hepatitis_conditions]['ALP'].value_counts()

ax2.hist(data_len_no_hepatitis, color='green')
ax2.set_title('Not Having Hepatitis')

fig.suptitle('Hepatitis Prediction')
plt.show()

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
#assuming 'classification' is a variable containing the target column name
classification = 'Category' #replace with your actual target column name
#select features (x) and target variable (y)
feature_columns = ['Unnamed: 0','Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
X = data[feature_columns]
y = data[classification]
X.replace('\t?',np.nan,inplace=True)
#convert columns to numeric (assuminf that they are numeric features)
X = X.apply(pd.to_numeric,errors='coerce')
#impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
#split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
# Create and fit the Logistic Regression model
model = LogisticRegression()
model.fit(train_X, train_Y)
# Make predictions on the test set
prediction = model.predict(test_X)
# Print accuracy and classification report
accuracy = metrics.accuracy_score(prediction,test_Y)
print('The accuracy of Logistic Regression is:', accuracy)
report = classification_report(test_Y, prediction)
print("Classification Report:\n", report)

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #linear regression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
prediction = model.predict(test_X)

# Assuming 'test_Y' contains the true labels for the test set
# Calculate the accuracy
accuracy = accuracy_score(test_Y, prediction.round())

# Print the accuracy
print('The accuracy of Linear Regression is:', accuracy)

#Evaluate the model using various metrices
mse = mean_squared_error(test_Y, prediction)
rmse = mean_squared_error(test_Y, prediction, squared=False) #Calculate the square root of MSE
mae = mean_absolute_error(test_Y, prediction)
r_squared=r2_score(test_Y, prediction)

print('Mean squared Error:',mse)
print('Root Mean Squared Error:',rmse)
print('Mean Absolute Error:',mae)
print('R-squared:',r_squared)
