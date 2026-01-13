import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def entrainer_et_tracer_clusters(k, liste_variables, dataframe, leg=''):
    """
    Entraîne un K-Means sur liste_variables et affiche les graphes.
    """
    # 1. Préparation des données pour l'entraînement
    X = dataframe[liste_variables].copy()

    encoder = OrdinalEncoder()
    X = encoder.fit_transform(X)
    

    # 2. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Entraînement du K-Means
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(X_scaled)

    # 4. Création des Subplots (2x2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if leg == '':
        fig.suptitle(f'K-Means avec k={k}\nVariables entraînées : {liste_variables}')
    else:
        fig.suptitle(f'K-Means avec k={k}\nVariables entraînées : {leg}')

    axes = axes.flatten()

    # Graphe 1 (en haut à gauche)
    axes[0].scatter(dataframe['Sleep_Hours_Per_Day'], dataframe['BMI'], c=clusters)
    axes[0].set_title("Sommeil vs BMI")

    # Graphe 2 (en haut à droite)
    axes[1].scatter(dataframe['Fast_Food_Meals_Per_Week'], dataframe['BMI'], c=clusters)
    axes[1].set_title("Fast Food vs BMI")

    # Graphe 3 (en bas à gauche)
    axes[2].scatter(dataframe['Age'], dataframe['BMI'], c=clusters)
    axes[2].set_title("Age vs BMI")

    # Graphe 4 (en bas à droite)
    axes[3].scatter(dataframe['Physical_Activity_Hours_Per_Week'], dataframe['BMI'], c=clusters)
    axes[3].set_title("Sport vs BMI")


    plt.tight_layout()
    


df_global = pd.read_csv('datas/fast_food_consumption_health_impact_dataset.csv')


vars = ['Fast_Food_Meals_Per_Week', 'Physical_Activity_Hours_Per_Week', 'BMI']
entrainer_et_tracer_clusters(k=3, liste_variables=vars, dataframe=df_global)

vars = ['Fast_Food_Meals_Per_Week', 'Physical_Activity_Hours_Per_Week', 'BMI']
entrainer_et_tracer_clusters(k=5, liste_variables=vars, dataframe=df_global)

vars = df_global.columns
entrainer_et_tracer_clusters(k=3, liste_variables=vars, dataframe=df_global, leg='Toutes les variables')


plt.show()