import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and preprocessing

#loading data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv('mail_data.csv')
#replace null value with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
# checking size of the dataframe
# mail_data.shape
# label spam mail as 0 an ham mail as 1.
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1
# separating the data as texts and label
x= mail_data['Message']
y= mail_data['Category']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=3)

# Feature Extraction
# transform the text data to feature vectors that can be used as input to the Logistic regression
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# convert y_train and y_test values as integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Logistic regression
model= LogisticRegression()

# training th elogistic regression model with the training data
model.fit(x_train_features,y_train)

# prediction on training data
prediction_on_training_data = model.predict(x_train_features)
model_accuracy_training = accuracy_score(y_train,prediction_on_training_data)

# prediction on test data
prediction_on_test_data = model.predict(x_test_features)
model_accuracy = accuracy_score(y_test,prediction_on_test_data)

input_mail = ["Click to get internship opprtunity"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
prediction
if prediction[0]==1:
    print('Ham mail')
else:
    print('Spam mail')