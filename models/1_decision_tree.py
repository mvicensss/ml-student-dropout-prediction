import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

target_col = 'Target'

#Separate predicting variables (X) and objective variables (y)
train_df = pd.read_csv('../data/processed/data_train.csv')
test_df = pd.read_csv('../data/processed/data_test.csv')

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Start model
clf_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
)

# Training
clf_tree.fit(X_train, y_train)

# Predict
y_pred = clf_tree.predict(X_test)

# Evaluation of metrics
print(f"Global Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report (focus on 'Recall' of Dropout class):")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf_tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_tree.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# Tree visualisation
plt.figure(figsize=(25, 15))
plot_tree(
    clf_tree,
    feature_names=X_train.columns,
    class_names=[str(c) for c in clf_tree.classes_],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=5,
)
plt.title("Decision Tree")
plt.show()
