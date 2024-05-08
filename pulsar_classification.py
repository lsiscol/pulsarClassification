# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle


# In this section, you can use a search engine to look for the functions that will help you implement the following steps

# Load dataset and show basic statistics
df = pd.read_csv('pulsar_stars.csv')

# 1. Show dataset size (dimensions)
print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Show what column names exist for the 9 attributes in the dataset
print(f"Column names: {df.columns.tolist()}")

# 3. Show the distribution of target_class column
print(df['target_class'].value_counts())

# 4. Show the percentage distribution of target_class column
print(df['target_class'].value_counts(normalize=True)*100)



# Separate predictor variables from the target variable (X and y as we did in the class)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]




# Create train and test splits for model development. Use the 80% and 20% split ratio
# Name them as X_train, X_test, y_train, and y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Standardize the features (Import StandardScaler here)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Remove this line after implementing train test split
X_test = scaler.transform(X_test) # Remove this line after implementing train test split

# Below is the code to convert X_train and X_test into data frames for the next steps
#cols = X_train.columns
#X_train = pd.DataFrame(X_train, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd
#X_test = pd.DataFrame(X_test, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd






# Train SVM with the following parameters.
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3
svm = SVC(kernel='rbf', C=10.0, gamma=0.3)
svm.fit(X_train, y_train)




# Test the above developed SVC on unseen pulsar dataset samples
y_pred = svm.predict(X_test)


# compute and print accuracy score
accuracy = svm.score(X_test, y_test)
print(f"Acuracy score: {accuracy:.3f}")





# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
filename = 'pulsarClassifier.sav'
pickle.dump(svm, open(filename, 'wb'))







# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
# Get and print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]



# Compute Precision and use the following line to print it
precision = precision_score(y_test, y_pred)
print('Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
recall = recall_score(y_test, y_pred)
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print('Specificity : {0:0.3f}'.format(specificity))

