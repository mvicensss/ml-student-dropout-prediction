import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay

target_col = 'Target'

train_df = pd.read_csv('../data/processed/data_train.csv')
test_df = pd.read_csv('../data/processed/data_test.csv')

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Start model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Training
log_reg.fit(X_train, y_train)

# Predict
y_pred = log_reg.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy}")
print(f"Classification Report: {classification_report(y_test,y_pred)}")

# Confussion Matrix
fig, ax = plt.subplots(figsize=(8,6))
ConfusionMatrixDisplay.from_estimator(log_reg,
                                       X_test,
                                       y_test,
                                       display_labels=log_reg.classes_,
                                       cmap='Blues',
                                       ax=ax
                                      )
ax.set_title('Confussion Matrix: Logistic regression')
plt.show()