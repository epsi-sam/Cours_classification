import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

df = pd.read_csv('datas/fast_food_consumption_health_impact_dataset.csv')


###############################################################
# Cas simple : on fait du clustering que sur deux variables :
###############################################################

df_reduit = df[['Fast_Food_Meals_Per_Week', 'BMI']]

# scaling
scaler = StandardScaler()
df_reduit = scaler.fit_transform(df_reduit)

##############################################
#### ELBOW et SILHOUETTE
##############################################

inertias = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(df_reduit)
    inertias.append(model.inertia_)
    labels = model.predict(df_reduit)
    score = silhouette_score(df_reduit, labels)
    silhouette_scores.append(score)

plt.figure()
plt.plot(K, inertias, marker='o')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.title('Méthode du coude (Elbow)')

plt.figure()
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Silhouette score')
plt.title('Silhouette score en fonction de k')



plt.show()