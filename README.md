# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

/*
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")

# Copy dataframe
df1 = df.copy()

# Drop unwanted columns
df1 = df1.drop(['sl_no', 'salary'], axis=1)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

categorical_cols = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

for col in categorical_cols:
    df1[col] = le.fit_transform(df1[col])

# Split features and target
x = df1.iloc[:, :-1]
y = df1['status']

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=8
)

# Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')

# Train model
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("\nConfusion Matrix (Numbers):\n", cm)
print("\nClassification Report:\n", cr)

# Confusion Matrix Heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Placed', 'Placed'],
    yticklabels=['Not Placed', 'Placed']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

Developed by: M.Hari Prasad
RegisterNumber:  (25013933)
*/
```

## Output:
<img width="474" height="695" alt="Screenshot 2026-02-12 131833" src="https://github.com/user-attachments/assets/7f6409e7-ac25-4c47-9f47-50acf77d350a" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
