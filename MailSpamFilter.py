import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

# Read the CSV file
df = pd.read_csv('mail_data.csv')
print(df)

# Handle missing values
data = df.where(pd.notnull(df), '')
data.head(10)
data.info()
data.shape

# Label encoding
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

# Feature and label selection
X = data['Message']
Y = data['Category']
print(X)
print(Y)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)

# TF-IDF vectorization
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert labels to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, Y_train)

prediction_on_training_data= model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Acc on training data : ', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

##Input Values from the mail
input_your_mail=['okay']
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)

print(prediction)

if(prediction[0]==1):
    print('Han mail')
else:
    print('Spam mail')
