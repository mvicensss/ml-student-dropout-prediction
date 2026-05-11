from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Modelo Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

# Entrenamiento
gb_model.fit(X_train, y_train)

# Predicción
y_pred_gb = gb_model.predict(X_test)

# Métricas
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb:.4f}")
print("Classification Report - Gradient Boosting:")
print(classification_report(y_test, y_pred_gb))

# Matriz de confusión
cm_gb = confusion_matrix(y_test, y_pred_gb, labels=gb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_gb, display_labels=gb_model.classes_)
disp.plot(cmap='Greens', values_format='d')
plt.title("Confusion Matrix - Gradient Boosting")
plt.show()