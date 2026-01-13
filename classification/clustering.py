import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv('datas/fast_food_consumption_health_impact_dataset.csv')

print(df.info())

print(df.describe())

###############################################################
# Cas simple : on fait du clustering que sur deux variables :
###############################################################

df_reduit = df[['Fast_Food_Meals_Per_Week', 'BMI']]

# scaling
scaler = StandardScaler()
df_reduit = scaler.fit_transform(df_reduit)

plt.scatter(df_reduit[:, 0], df_reduit[:, 1])
plt.xlabel('Fast Food')
plt.ylabel('BMI')
plt.title('Raw data)')

#### k = 3 ####

model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(df_reduit)

# Visualisation
plt.figure()
plt.scatter(df_reduit[:, 0], df_reduit[:, 1], c=clusters)
plt.xlabel('Fast Food')
plt.ylabel('BMI')
plt.title('k = 3')

#### k = 5 ####

model = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = model.fit_predict(df_reduit)

# Visualisation 
plt.figure()
plt.scatter(df_reduit[:, 0], df_reduit[:, 1], c=clusters)
plt.xlabel('Fast Food')
plt.ylabel('BMI')
plt.title('k = 5')


###############################################################
# Cas quatre variables
###############################################################

df_reduit = df[['Fast_Food_Meals_Per_Week', 'BMI', 'Average_Daily_Calories', 'Overall_Health_Score']]

# scaling
scaler = StandardScaler()
df_reduit = scaler.fit_transform(df_reduit)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(df_reduit)

# Visualisation 
plt.figure()
scatter = plt.scatter(df_reduit[:, 0], df_reduit[:, 1], c=clusters)
handles, labels = scatter.legend_elements()

plt.legend(handles, labels, title="Clusters")
plt.xlabel('Fast Food')
plt.ylabel('BMI')
plt.title('k = 3, quatre variables')
plt.show()

