import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("data.csv")

print("First 5 rows:")
print(df.head())

print("\nMissing values before cleaning:")
print(df.isnull().sum())


# 2️⃣ Handle missing values properly

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())


# 3️⃣ Convert target variable (Y/N → 1/0)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})


# 4️⃣ Convert categorical variables to numeric
df = pd.get_dummies(df, drop_first=True)


# 5️⃣ Split features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']


# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7️⃣ Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# 8️⃣ Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)


# 9️⃣ Save model
joblib.dump(model, "loan_model.pkl")

print("\nModel saved successfully!")