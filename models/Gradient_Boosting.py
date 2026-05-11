from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# Load processed data
train_df = pd.read_csv('../data/processed/data_train.csv')
test_df  = pd.read_csv('../data/processed/data_test.csv')

target_col = 'Target'

# Initiate model
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

# Training
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

gb_model.fit(X_train, y_train)

# Predict
y_pred_gb = gb_model.predict(X_test)

# Evaluate metrics
print(f"\nGlobal Accuracy (Gradient Boosting): {accuracy_score(y_test, y_pred_gb):.4f}\n")
print("Classification Report - Gradient Boosting:")
print(classification_report(y_test, y_pred_gb))

# Confusion Matrix
cm_gb = confusion_matrix(y_test, y_pred_gb, labels=gb_model.classes_)
disp_gb = ConfusionMatrixDisplay(confusion_matrix=cm_gb, display_labels=gb_model.classes_)

disp_gb.plot(cmap='Greens', values_format='d')
plt.title("Confusion Matrix - Gradient Boosting")
plt.show()