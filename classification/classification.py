import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn import svm
import matplotlib.pyplot as plt


df = pd.read_csv('datas/fast_food_consumption_health_impact_dataset.csv')

print(df.info())

print(df.describe())

df = df.drop(columns=['Gender'])

# Remaplace Yes et No par 1 et 0
mapping = {'Yes': 1, 'No': 0}
target = df['Digestive_Issues'].map(mapping)

# Drop de la cible
df = df.drop(columns=['Digestive_Issues'])

# scaling
scaler = StandardScaler()
df = scaler.fit_transform(df)

# Découpage (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

#####################################
# Reg logistique
#####################################

# Création et entraînement
# class_weight permet de prendre en compte le déséquilibre des classes
# Donne un poids plus fort aux classes minoritaires
log_model = LogisticRegression(class_weight='balanced')
log_model.fit(X_train, y_train)

# Prédiction
y_pred_log = log_model.predict(X_test)

print("Régression Logistique :\n")
print("Matrice de confusion : ")
print(confusion_matrix(y_test, y_pred_log))
print("Métriques : ")
print(classification_report(y_test, y_pred_log))
print(f"AUC = {roc_auc_score(y_test, y_pred_log)}")

#####################################
# SVM
#####################################

# Création et entraînement

clf = svm.SVC(class_weight='balanced')
clf.fit(X_train, y_train)

y_pred_svm = clf.predict(X_test)

print("SVM")
print("Matrice de confusion : ")
print(confusion_matrix(y_test, y_pred_svm))
print("Métriques : ")
print(classification_report(y_test, y_pred_svm))
print(f"AUC = {roc_auc_score(y_test, y_pred_svm)}")


#####################################
# ROC
#####################################


fig, ax = plt.subplots(figsize=(8, 6))

# Tracer la courbe pour la Régression Logistique
RocCurveDisplay.from_estimator(log_model, X_test, y_test, ax=ax, name='Régression Logistique')

# Tracer la courbe pour la SVM
RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, name='SVM')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Hasard (AUC = 0.50)')
plt.title("Courbe ROC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()