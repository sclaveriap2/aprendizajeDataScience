```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar dataset
iris = load_iris()
X, y = iris.data, iris.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

```

    Precisión del modelo: 1.00
    


```python
print("Predicciones del modelo:")
print(y_pred)

# Mostrar las especies reales del conjunto de prueba
print("Especies reales del conjunto de prueba:")
print(y_test)
```

    Predicciones del modelo:
    [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
    Especies reales del conjunto de prueba:
    [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
    


```python
import pandas as pd

species_names = iris.target_names  # Nombres de las especies

# Imprimir predicciones y resultados reales
print("Predicciones del modelo (especies):")
print([str(species_names[i]) for i in y_pred])

print("\nEspecies reales del conjunto de prueba:")
print([str(species_names[i]) for i in y_test])

# Contar cuántas veces se predijo cada especie
predicted_counts = pd.Series([str(species_names[i]) for i in y_pred]).value_counts()
real_counts = pd.Series([str(species_names[i]) for i in y_test]).value_counts()

print("\nCantidad de predicciones por especie:")
print(predicted_counts)

print("\nCantidad de especies reales en el conjunto de prueba:")
print(real_counts)
```

    Predicciones del modelo (especies):
    ['versicolor', 'setosa', 'virginica', 'versicolor', 'versicolor', 'setosa', 'versicolor', 'virginica', 'versicolor', 'versicolor', 'virginica', 'setosa', 'setosa', 'setosa', 'setosa', 'versicolor', 'virginica', 'versicolor', 'versicolor', 'virginica', 'setosa', 'virginica', 'setosa', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'setosa', 'setosa']
    
    Especies reales del conjunto de prueba:
    ['versicolor', 'setosa', 'virginica', 'versicolor', 'versicolor', 'setosa', 'versicolor', 'virginica', 'versicolor', 'versicolor', 'virginica', 'setosa', 'setosa', 'setosa', 'setosa', 'versicolor', 'virginica', 'versicolor', 'versicolor', 'virginica', 'setosa', 'virginica', 'setosa', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'setosa', 'setosa']
    
    Cantidad de predicciones por especie:
    virginica     11
    setosa        10
    versicolor     9
    Name: count, dtype: int64
    
    Cantidad de especies reales en el conjunto de prueba:
    virginica     11
    setosa        10
    versicolor     9
    Name: count, dtype: int64
    


```python

```
