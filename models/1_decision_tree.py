import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


target_col = 'Target'

#Separate predicting variables (X) and objective variables (y)
train_df = pd.read_csv('data/processed/data_train.csv')
test_df = pd.read_csv('data/processed/data_test.csv')

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

#start model
custom_weights={
    'Dropout': 3.0,
    'Enrolled': 1.5,
    'Graduate': 1.0
}

clf_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    #class_weight={'Dropout': 4.0, 'Enrolled': 1.0, 'Graduate': 1.0},
    class_weight=custom_weights,
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

#Tree visualisation
plt.figure(figsize=(20, 10))
plot_tree(
    clf_tree,
    feature_names=X_train.columns,
    class_names=[str(c) for c in clf_tree.classes_],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3,
)
plt.title("Decision Tree")
plt.show()