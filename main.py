from neuralnetworkmultilayer import NeuralNetwork
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle as pkl
import matplotlib.pyplot as plt

# importo il dataset
data = pd.read_csv("datasets/padova_lag.csv")

# seleziono y e X, e li divido in dataset di addestramento e di test
y = data["precipBin"]
X = data.drop(columns=["date_time","precipBin"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

# standardizzo X
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

# funzione che calcola l'accuratezza
def score(model, X, y):
    global X_test_std, y_test
    y_pred = model.predict(X_test_std)
    return (np.sum(y_test == y_pred)).astype(np.float32) / len(y_test)

param_grid = {"n_hidden_units" : [[200,100,50]],
              "eta" : [0.001, 0.0001, 0.00001],
              "l2" : [0.0001, 0.00001, 0.000001],
              "epochs" : [30]}

gs = GridSearchCV(estimator=NeuralNetwork(),
                  param_grid=param_grid,
                  scoring=score, n_jobs=-1, cv=3, verbose=3)

###################################################################
## questa parte di codice richiede molto tempo di esecuzione       
## può essere saltata rimuovendo gli '#' prima delle tre virgolette
"""
gs.fit(X_train_std, y_train)
print(gs.best_params_)
with open("gs.pkl", "wb") as file:
    pkl.dump(gs, file)
"""
###################################################################

with open("gs.pkl", "rb") as file:
    gs = pkl.load(file)

# risultati ricerca a griglia
results = np.reshape(gs.cv_results_["mean_test_score"], newshape=(3,3))
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(results, cmap=plt.cm.Blues, alpha=0.3)
for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        ax.text(x=j, y=i, s=round(results[i, j]*100,3), va='center', ha='center')
plt.xticks([0,1,2], [0.0001, 0.00001, 0.000001])
plt.yticks([0,1,2], [0.001, 0.0001, 0.00001])
plt.title("Grid Search results")
plt.xlabel('L2 regulation')
plt.ylabel('Eta')
plt.tight_layout()
plt.show()

rn = gs.best_estimator_

# miglior modello
print(f"Best params:\n{gs.best_params_}")

# accuracy plot
rn.plot_accuracy("both", save=True)

# matrice di confusione
y_test_pred = rn.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.title("Neural network")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks([0,1], ["no precipitazioni", "precipitazioni"])
plt.yticks([0,1], ["no precipitazioni", "precipitazioni"])
plt.tight_layout()
plt.show()

# metirche di valutazione
print("accuracy:", score(rn, X_test_std, y_test))
print("sensibility:",confmat[1,1]/(confmat[1,0]+confmat[1,1]))
print("specificity:",confmat[0,0]/(confmat[0,1]+confmat[0,0]))