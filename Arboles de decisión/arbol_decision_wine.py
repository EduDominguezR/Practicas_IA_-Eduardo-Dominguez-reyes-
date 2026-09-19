"""
Clasificación con Árbol de Decisión - Wine Dataset
Práctica: Entrenamiento e interpretación de un árbol de decisión
usando el dataset clásico de vinos incluido en scikit-learn.
"""

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Cargar el dataset del vino
wine = load_wine()
X, y = wine.data, wine.target

print("Muestras totales:", X.shape[0])
print("Características por muestra:", X.shape[1])
print("Clases:", set(y))
print("Nombres de características:", wine.feature_names)
print("-" * 60)

# 2. Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Función auxiliar para entrenar, evaluar y mostrar reglas
def entrenar_y_evaluar(max_depth):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== max_depth = {max_depth} ===")
    print(f"Profundidad real alcanzada: {tree.get_depth()}")
    print(f"Número de hojas: {tree.get_n_leaves()}")
    print(f"Precisión en datos de prueba: {acc:.4f}")
    print("Reglas del árbol:")
    print(export_text(tree, feature_names=list(wine.feature_names)))
    return tree, acc


# 4. Entrenar con distintos valores de max_depth (2, 3, 4, 5 y sin límite)
for profundidad in [2, 3, 4, 5, None]:
    entrenar_y_evaluar(profundidad)

# 5. Modelo final recomendado (max_depth=2, más interpretable)
print("\n" + "=" * 60)
print("MODELO FINAL (max_depth=2, interpretable)")
tree_final, acc_final = entrenar_y_evaluar(2)
print(f"\nPrecisión final del modelo: {acc_final * 100:.2f}%")
