# -----------------------------------------------
# TITANIC SURVIVAL PREDICTION - LOGISTIC REGRESSION
# -----------------------------------------------

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
train = pd.read_csv("train.csv")   
test = pd.read_csv("test.csv")

# Step 3: Select useful features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features].copy() 
y = train['Survived']

# Step 4: Handle Missing Values safely
X.loc[:, 'Age'] = X['Age'].fillna(X['Age'].median())
X.loc[:, 'Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

# Step 5: Encode Categorical Data
le_sex = LabelEncoder()
le_emb = LabelEncoder()
X.loc[:, 'Sex'] = le_sex.fit_transform(X['Sex'])
X.loc[:, 'Embarked'] = le_emb.fit_transform(X['Embarked'])

# Step 6: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build and Train Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Step 8: Predictions and Evaluation
y_pred = model.predict(X_val)

print("✅ Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Step 9: Predict on Test Data for Submission
test = test.copy()  # Make a copy to avoid warnings

# Fill missing values in test set safely
test.loc[:, 'Age'] = test['Age'].fillna(X['Age'].median())
test.loc[:, 'Fare'] = test['Fare'].fillna(X['Fare'].median())

# Encode categorical columns using the same encoders
test.loc[:, 'Sex'] = le_sex.transform(test['Sex'])
test.loc[:, 'Embarked'] = le_emb.transform(test['Embarked'])

# Prepare test features and predict
test_features = test[features]
test_preds = model.predict(test_features)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_preds
})

submission.to_csv("titanic_logreg_submission.csv", index=False)
print("\n🎯 Submission file saved as titanic_logreg_submission.csv")
