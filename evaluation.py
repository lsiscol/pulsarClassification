# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle


# 2. Create test set if you like to do the 80:20 split programmatically or if you have not already split the data at this point
df = pd.read_csv('pulsar_stars.csv')
X = df.drop('target_class', axis=1)
y = df['target_class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

# 3. Load your saved model for pulsar classifier that you saved in pulsar_classification.py via Pikcle
with open('pulsarClassifier.sav', 'rb') as file:
    pulsar_classifier = pickle.load(file)

# 4. Make predictions on test_set created from step 2
X_test_std = scaler.transform(X_test)
y_pred = pulsar_classifier.predict(X_test_std)

# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)


precision = precision_score(y_test, y_pred, zero_division=1)
print("Precision Score:", precision)


recall = recall_score(y_test, y_pred)
print("Recall Score:", recall)

#TN, FP, FN, TP = cm.ravel()
#if TN + FP == 0:
    #specificity = 1
#else:
    #specificity = TN / (TN + FP)
specificity = cm[0,0]/(cm[0,0]+cm[0,1])
print("Specificity:", specificity)


# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)

#print the scores of accuracy, precision, recall,  and specificity
print("Accuracy Score:", accuracy)
print("Precision Score:", precision)
print("Recall Score:", recall)
print("Specificity:", specificity)