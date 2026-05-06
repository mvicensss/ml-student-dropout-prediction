from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def knn_comparacion(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, k=5):
    # -------- SIN ESCALADO --------
    knn_no = KNeighborsClassifier(n_neighbors=k)
    knn_no.fit(X_train, y_train)
    y_pred_no = knn_no.predict(X_test)
    acc_no = accuracy_score(y_test, y_pred_no)

    print("=== KNN SIN ESCALADO ===")
    print(f"Accuracy: {acc_no:.3f}")
    print(classification_report(y_test, y_pred_no))

    ConfusionMatrixDisplay.from_estimator(
        knn_no, X_test, y_test, cmap="Reds"
    )
    plt.title("Confusion Matrix - KNN sin escalado")
    plt.show()

    # -------- CON ESCALADO --------
    knn_scaled = KNeighborsClassifier(n_neighbors=k)
    knn_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = knn_scaled.predict(X_test_scaled)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)

    print("\n=== KNN CON ESCALADO ===")
    print(f"Accuracy: {acc_scaled:.3f}")
    print(classification_report(y_test, y_pred_scaled))

    ConfusionMatrixDisplay.from_estimator(
        knn_scaled, X_test_scaled, y_test, cmap="Blues"
    )
    plt.title("Confusion Matrix - KNN con escalado")
    plt.show()

    # -------- RESUMEN --------
    print("\n=== RESUMEN ===")
    print(f"Accuracy sin escalado: {acc_no:.3f}")
    print(f"Accuracy con escalado: {acc_scaled:.3f}")

# LLAMADA (una sola línea en el notebook)
knn_comparacion(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
