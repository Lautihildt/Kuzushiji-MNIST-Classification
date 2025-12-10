# Trabajo Práctico 02: Clasificación de Kuzushiji-MNIST
# Laboratorio de Datos

# Grupo: "Grupo 1"
# Integrantes:
#   - Hildt, Lautaro (LU: 1166/24)
#   - Fisanotti, Juan Bautista (LU: 1146/24)

# Contenido:
# Este script de Python contiene todo el código para el análisis del TP2.
# El script está dividido en las siguientes secciones:

# 1. Ejercicio 1: Análisis Exploratorio
#    - Mapas de calor de desviación estándar.
#    - Matriz de distancia euclideana entre clases.

# 2. Ejercicio 2: Clasificación Binaria
#    - Comparación de 3 estrategias de selección de atributos (con seed 0 y 1).
#    - Optimización de hiperparámetros N y k.

# 3. Ejercicio 3: Clasificación Multiclase (Árboles de Decisión)
#    - Análisis de sensibilidad de `max_depth` (split simple).
#    - Búsqueda de hiperparámetros (`criterion`, `max_depth`, `min_samples_split`) con GridSearchCV y K-Fold.
#    - Análisis de sensibilidad de `min_samples_split` (K-Fold).
#    - Evaluación del modelo final sobre el conjunto held-out (Matriz de Confusión).
#%%
#Importamos todas las librerias que usaremos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

#Herramientas de Scikit-learn
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report

# %% 1. Cargamos los datos

df = pd.read_csv("kuzushiji_full.csv")
class_map = pd.read_csv("kmnist_classmap_char.csv")
imagenes = df.iloc[:, 0:784].values

#%% Análisis Exploratorio

# Mapa de calor de desviación estándar (Todas las clases)

pixel_var = imagenes.std(axis=0).reshape(28, 28)

plt.figure(figsize=(6, 5))
plt.imshow(pixel_var, cmap='hot')
plt.grid(False)
plt.colorbar()
plt.title("Desviación estándar por píxel (Todas las clases)")
plt.show()

# Matriz de distancia euclideana entre clases
imagenes_promedio = []
for i in range(10):
    imgs_clase = df[df['label']==i].iloc[:, :784].values
    img_prom = imgs_clase.mean(axis=0)
    imagenes_promedio.append(img_prom)

promedios_flat = np.array(imagenes_promedio)
distancias = cdist(promedios_flat, promedios_flat, metric='euclidean')

plt.figure(figsize=(8, 6))
sns.heatmap(distancias, annot=True, fmt=".1f", cmap='viridis')
plt.title("Distancia euclidiana entre imágenes promedio de cada clase")
plt.xlabel("Clase")
plt.ylabel("Clase")
plt.show()

# Mapa de calor de desviación estándar de la clase 8
imagenes_c8 = df[df['label'] == 8].iloc[:, 0:784].values
desviacion_c8 = np.std(imagenes_c8, axis=0).reshape(28, 28)

plt.figure(figsize=(6, 5))
plt.imshow(desviacion_c8, cmap='hot')
plt.grid(False)
plt.colorbar()
plt.title("Desviación estándar por píxel (Clase 8)")
plt.show()

# %% 2. Ejercicio 2: Clasificación Binaria (KNN, Clases 4 vs 5)

#Filtramos y dividimos en train/test
df_45 = df[df['label'].isin([4,5])].copy()

X_knn = df_45.iloc[:, :784].values
y_knn = df_45['label'].values

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn, y_knn, test_size=0.2, random_state=42, stratify=y_knn)

#Calculamos los píxeles que son más distintos

imagenes_4 = X_train_knn[y_train_knn == 4]
imagenes_5 = X_train_knn[y_train_knn == 5]
promedio_4 = imagenes_4.mean(axis=0)
promedio_5 = imagenes_5.mean(axis=0)
diferencia_abs = np.abs(promedio_4 - promedio_5)

df_diff = pd.DataFrame({
    'pixel': np.arange(X_train_knn.shape[1], dtype=int),
    'diferencia_abs': diferencia_abs
}).sort_values(by='diferencia_abs', ascending=False)
top100_pixels = df_diff.head(100)['pixel'].astype(int).values

#Comparación de 3 Estrategias vs N (con seed 0 y 1)
valores_N = [1, 3, 8, 10, 15, 20]

for seed_val in [0, 1]:
    resultados_lineas = []
    np.random.seed(seed_val) 

    for N in valores_N:
        # Estrategia 1: "Mejores N"
        idx_diff = df_diff.head(N)['pixel'].astype(int).values
        
        # Estrategia 2: "Aleatorios del Top 100"
        idx_rand100 = np.random.choice(top100_pixels, size=N, replace=False)
        
        # Estrategia 3: "Aleatorios"
        idx_rand_all = np.random.choice(np.arange(784), size=N, replace=False)
        
        estrategias = {
            "Mejores N píxeles": idx_diff,
            "N píxeles aleatorios del top 100": idx_rand100,
            "N píxeles aleatorios": idx_rand_all
        }

        for nombre, idx in estrategias.items():
            scaler = StandardScaler()
            X_train_r = scaler.fit_transform(X_train_knn[:, idx])
            X_test_r  = scaler.transform(X_test_knn[:, idx])

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train_r, y_train_knn)
            # Usamos _knn para predecir y evaluar
            y_pred = knn.predict(X_test_r)
            acc = accuracy_score(y_test_knn, y_pred)
            resultados_lineas.append({"N": N, "Estrategia": nombre, "Accuracy": acc})

    #Graficamos el resultado para este valor de seed
    df_lineas = pd.DataFrame(resultados_lineas)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_lineas, x='N', y='Accuracy', hue='Estrategia', marker='o', linewidth=2)
    plt.title(f"Impacto de la Selección de Atributos (seed={seed_val}, k=3)", fontsize=16)
    plt.xlabel("Cantidad de atributos (N)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(valores_N)
    plt.ylim(0, 1)
    plt.legend(title='Estrategia de Selección')
    plt.tight_layout()
    plt.show()

#Evaluamos distintos valores de N y k mediante Heatmap de la mejor estrategia

#Basado en el análisis anterior, optimizamos la estrategia 
#N píxeles aleatorios del top 100 con seed=0

valores_N_hm = [1, 3, 8, 10, 15, 20]
valores_k = [1, 3, 5, 7, 9]

resultados_hm = []
np.random.seed(0) 

for N in valores_N_hm:
    idx_heatmap = np.random.choice(top100_pixels, size=N, replace=False)
    
    scaler = StandardScaler()
    X_train_r = scaler.fit_transform(X_train_knn[:, idx_heatmap])
    X_test_r  = scaler.transform(X_test_knn[:, idx_heatmap])

    for k in valores_k:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_r, y_train_knn)
        # Usamos _knn para predecir y evaluar
        y_pred = knn.predict(X_test_r)
        acc = accuracy_score(y_test_knn, y_pred)
        resultados_hm.append({'N': N, 'k': k, 'Accuracy': acc})

df_hm = pd.DataFrame(resultados_hm)

# Gráfico Heatmap
tabla_acc_hm = df_hm.pivot(index='N', columns='k', values='Accuracy')
plt.figure(figsize=(8,6))
sns.heatmap(tabla_acc_hm, annot=True, fmt=".3f", cmap="Blues")
plt.title("Accuracy según N y k (Estrategia 'Aleatorios Top 100', seed=0)", fontsize=14)
plt.ylabel("Cantidad de atributos (N)")
plt.xlabel("Número de vecinos (k)")
plt.gca().invert_yaxis() #Acomodamos para que N crezca hacia arriba
plt.show()

# %% Clasificación Multiclase mediante arboles de decision

#Separamos en desarrollo (dev) y validación final (held-out)

X_total = df.iloc[:, :784].values
y_total = df['label'].values

X_desarrollo, X_validacion_final, y_desarrollo, y_validacion_final = train_test_split(
    X_total,
    y_total,
    test_size=0.2,
    random_state=42,
    stratify=y_total
)

#Probamos con distintas profundidades (split simple)

# Dividimos el conjunto de desarrollo en entrenamiento y validación interna
X_entrenamiento_dev, X_validacion_dev, y_entrenamiento_dev, y_validacion_dev = train_test_split(
    X_desarrollo,
    y_desarrollo,
    test_size=0.2, # 20% del 80% original
    random_state=42,
    stratify=y_desarrollo
)

resultados_prof = []
profundidades = range(1, 11)

for p in profundidades:
    arbol_simple = DecisionTreeClassifier(max_depth=p, criterion='gini', random_state=42)
    arbol_simple.fit(X_entrenamiento_dev, y_entrenamiento_dev)
    
    y_pred_simple = arbol_simple.predict(X_validacion_dev)
    accuracy = accuracy_score(y_validacion_dev, y_pred_simple)
    
    resultados_prof.append({'Profundidad': p, 'Accuracy': accuracy})

df_prof = pd.DataFrame(resultados_prof)

plt.figure(figsize=(8, 5))
sns.lineplot(data=df_prof, x='Profundidad', y='Accuracy', marker='o')
plt.title('Accuracy (validación simple) vs. Profundidad del Árbol')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy')
plt.xticks(profundidades)
plt.tight_layout()
plt.show()

#Buscamos los mejores Hiperparámetros con K-Fold (GridSearchCV)
param_grid_main = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [10]
}

cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)

#Renombramos a 'gs_main' para no confundir con el de sensibilidad
gs_main = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_main,
    scoring='accuracy', 
    refit='accuracy',   
    cv=cv_kfold        
)

gs_main.fit(X_desarrollo, y_desarrollo)


#Sensibilidad a 'min_samples_split'
param_grid_sens = {
    'min_samples_split': [5, 10, 20, 50, 100, 200, 350, 500],
    'max_depth': [10], 
    'criterion': ['gini']
}

gs_sens = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_sens,
    scoring='accuracy',
    cv=cv_kfold
)

gs_sens.fit(X_desarrollo, y_desarrollo)

# Pasamos a DataFrame
cv_results_df_sens = pd.DataFrame(gs_sens.cv_results_)
cv_results_df_sens['min_samples_split'] = cv_results_df_sens['params'].apply(lambda d: d['min_samples_split'])

# Gráficamos
labels_eje_x = list(map(str, param_grid_sens['min_samples_split']))
plt.figure(figsize=(8,5))
sns.lineplot(
    data=cv_results_df_sens,
    x='min_samples_split',
    y='mean_test_score',
    marker='o'
)
plt.xscale('log')
plt.title("Sensibilidad del modelo a min_samples_split (CV 5-fold)")
plt.xticks(param_grid_sens['min_samples_split'], labels=labels_eje_x)
plt.xlabel("min_samples_split (escala logarítmica)")
plt.ylabel("Accuracy promedio (CV 5-fold)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Evaluamos el mejor modelo en el conjunto held-out
mejores_parametros = gs_main.best_params_

# Entrenamos el mejor modelo con TODO el conjunto de desarrollo
clf_mejor = DecisionTreeClassifier(
    criterion=mejores_parametros['criterion'],
    max_depth=mejores_parametros['max_depth'],
    min_samples_split=mejores_parametros['min_samples_split'],
    random_state=42
)
clf_mejor.fit(X_desarrollo, y_desarrollo)

y_pred_validacion_final = clf_mejor.predict(X_validacion_final)

# Matriz de confusión
cm = confusion_matrix(y_validacion_final, y_pred_validacion_final, normalize='true')

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=True, linewidths=0.5, square=True)
plt.title("Matriz de confusión - Mejor modelo (held-out)")
plt.xlabel("Etiqueta predicha")
plt.ylabel("Etiqueta real")
plt.tight_layout()
plt.show()
