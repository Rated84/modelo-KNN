import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from ucimlrepo import fetch_ucirepo 

wine_quality = fetch_ucirepo(id=186) 

df = pd.read_csv('wine+quality/winequality-red.csv', sep=';')

def classificar(q):
    if q <= 5: return 'ruim'
    elif q <= 7: return 'medio'
    else: return 'bom'

df['classe'] = df['quality'].apply(classificar)

X = df.drop(columns=['quality', 'classe']).values
y = df['classe'].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2 ,random_state=42, stratify=y
)

print(pd.Series(y).value_counts().sort_index())


ks = list(range(1,50))
medias = []

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    medias.append(scores.mean())
    print(f"k={k:2d} → acurácia média: {scores.mean():.4f} ± {scores.std():.4f}")

# gráfico 
plt.plot(ks, medias, marker='o')
plt.xlabel('k')
plt.ylabel('Acurácia média (5-fold CV)')
plt.title('Escolha do k — Validação Cruzada')
plt.xticks(ks[::2])
plt.grid(True)
plt.tight_layout()
plt.savefig("resultado.png", dpi=200, bbox_inches="tight")
plt.close()


best_k = ks[np.argmax(medias)]
print(f"\nMelhor k: {best_k}")

modelo_final = KNeighborsClassifier(n_neighbors=best_k)
modelo_final.fit(X_train, y_train)

y_prev = modelo_final.predict(X_test)

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_prev))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_prev,zero_division = 0, target_names=['bom', 'medio', 'ruim']))

