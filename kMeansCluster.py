#Import library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Membaca data
data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/RFM_customer.csv", encoding='utf8')
RFM_km = data.drop(["customer_id"], axis=1)

standard_scaler = StandardScaler()
RFM_standardized = standard_scaler.fit_transform(RFM_km)
RFM_standardized = pd.DataFrame(RFM_standardized)
RFM_standardized.columns = ["Frequency","Monetary","Recency"]

#Mengatur parameter k-Means
k_means4 = KMeans(n_clusters=4, random_state=0)
k_means5 = KMeans(n_clusters=5, random_state=0)

#Menjalankan algoritma k-means dengan jumlah cluster = 4
k_means4.fit(RFM_standardized)

#Pred menyimpan hasil prediksi label cluster untuk setiap data dengan jumlah cluster = 4
pred = k_means4.predict(RFM_standardized)

#Menggabungkan dataframe data dan hasil label clustering
RFM_labeled = pd.concat([RFM_standardized, pd.Series(pred).rename("cluster")], axis=1)

#Menampilkan hasil clustering untuk setiap data dalam bentuk boxplot
fig, ax = plt.subplots(1,3, figsize=(18,10))
sns.boxplot(x="cluster", y="Recency", data=RFM_labeled, ax=ax[0])
sns.boxplot(x="cluster", y="Frequency", data=RFM_labeled, ax=ax[1])
sns.boxplot(x="cluster", y="Monetary", data=RFM_labeled, ax=ax[2])
plt.suptitle("Clustering dengan 4 cluster", fontsize=16)
plt.show()

#Menjalankan algoritma k-means dengan jumlah cluster = 5
k_means5.fit(RFM_standardized)

#Pred menyimpan hasil prediksi label cluster untuk setiap data dengan jumlah cluster = 5
pred = k_means5.predict(RFM_standardized)

#Menggabungkan RFM dan hasil label clustering
RFM_labeled = pd.concat([RFM_standardized, pd.Series(pred).rename("cluster")], axis=1)

#Menampilkan hasil clustering untuk setiap data dalam bentuk boxplot
fig, ax = plt.subplots(1,3, figsize=(18,10))
sns.boxplot(x="cluster", y="Recency", data=RFM_labeled, ax=ax[0])
sns.boxplot(x="cluster", y="Frequency", data=RFM_labeled, ax=ax[1])
sns.boxplot(x="cluster", y="Monetary", data=RFM_labeled, ax=ax[2])
plt.suptitle("Clustering dengan 5 cluster", fontsize=16)
plt.show()