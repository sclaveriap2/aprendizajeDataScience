```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Cargar dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Convertir etiquetas a enteros (por defecto están como strings)
y = y.astype(int)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir hiperparámetros a probar
param_dist = {
    'n_estimators': randint(50, 300),     # Número de árboles (aleatorio entre 50 y 300)
    'max_depth': [10, 20, 30, None],      # Profundidad del árbol
    'min_samples_split': randint(2, 10),  # Min muestras para dividir
    'min_samples_leaf': randint(1, 4),    # Min muestras en hoja
    'max_features': ['sqrt', 'log2']      # Cantidad de features evaluadas
}

# Crear búsqueda de mejor hiperparámetro
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=40), param_distributions=param_dist, 
                                   n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42) 
random_search.fit(X_train, y_train) # ✅ Ahora X_train y y_train están definidos

# Imprimir el mejor valor de n_estimators
print(f"Mejores hiperparámetros encontrados: {random_search.best_params_}")
```

    Fitting 3 folds for each of 20 candidates, totalling 60 fits
    Mejores hiperparámetros encontrados: {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 216}
    


```python
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Convertir etiquetas a enteros (por defecto están como strings)
y = y.astype(int)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=216,
                               max_depth= 30,
                               max_features='sqrt',
                               min_samples_leaf= 1,
                               min_samples_split= 2,)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")
```

    Precisión del modelo: 0.97
    
