# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV File
file_path = '/content/spam.csv'  # Update with the correct file path
try:
    df = pd.read_csv(file_path)
    print("Successfully read the CSV file.")
except UnicodeDecodeError:
    print("Failed to read the CSV file with utf-8 encoding.")
    try:
        # Try reading with a different encoding
        df = pd.read_csv(file_path, encoding='latin1')
        print("Successfully read the CSV file with latin1 encoding.")
    except UnicodeDecodeError:
        print("Failed to read the CSV file with latin1 encoding. Please check the file encoding.")

# Print the first few rows to verify the data is read correctly
print(df.head())

# Droping the unwanted Columns
columns_to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]  # Corrected column names
df.drop(columns=columns_to_drop, inplace=True)

# Renaming the Columns "v1" and "v2" to New Names
new_column_names = {"v1": "Category", "v2": "Message"}
df.rename(columns=new_column_names, inplace=True)

# Replace any NaN Values in the DataFrame with a Space
data = df.where((pd.notnull(df)), ' ')

# Converting the "Category" Column Values to Numerical Representation (0 for "SPAM" and 1 for "HAM")
data.loc[data["Category"] == "spam", "Category"] = 0
data.loc[data["Category"] == "ham", "Category"] = 1

# Separate the Feature (Message) and Target (Category) Data
X = data["Message"]
Y = data["Category"]

# Split the Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Create a TF-IDF Vectorizer to Convert Text Message into Numerical Features
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

# Create the Training and Testing Text Message into Numerical Features Using TF-IDF
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Create the Target Values to Integers (0 and 1)
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

# Using LogisticRegression Model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Prediction on the Training Data and Calculate the Accuracy
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print("Accuracy on Training Data:", accuracy_on_training_data)

# Prediction on the Testing Data and Calculate the Accuracy
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print("Accuracy on Testing Data:", accuracy_on_test_data)

# Test the Model with Some Custom Email Messages
input_your_mail = ["Congratulations! You Have Won a Free Vacation to an Exotic Destination. Click the Link to Claim Your Prize"]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)

# Prediction Result
if prediction[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")

input_your_mail = ["Meeting Reminder: Tomorrow, 10 AM, Conference Room. See You There!"]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)

# Prediction Result
if prediction[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")

# Data Visualization Distribution of Spam and Ham Emails
spam_count = data[data['Category'] == 0].shape[0]
ham_count = data[data['Category'] == 1].shape[0]

plt.bar(['Spam', 'Ham'], [spam_count, ham_count])
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Spam and Ham Emails')
plt.show()

# Confusion Matrix
cm = confusion_matrix(Y_test, prediction_on_test_data)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap='Oranges', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

probabilities = model.predict_proba(X_test_features)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, probabilities)
roc_auc = roc_auc_score(Y_test, probabilities)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
