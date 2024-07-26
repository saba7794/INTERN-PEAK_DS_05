#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Load the dataset
file_path = 'Online Retail.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Display the first few rows of the dataset
df.head()


# In[4]:


# Handle missing values
df.dropna(subset=['CustomerID'], inplace=True)

# Remove unnecessary columns
df = df[['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']]

# Create TotalAmount column
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Aggregate purchase history per customer
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'TotalAmount': 'sum'  # Monetary
}).reset_index()

# Rename columns
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Display the first few rows of the RFM dataframe
rfm.head()


# In[5]:


from sklearn.preprocessing import StandardScaler

# Scale the RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Convert the scaled data back to a dataframe
rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

# Display the first few rows of the scaled RFM dataframe
rfm_scaled.head()


# In[6]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determine the optimal number of clusters using the Elbow Method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

# Plot the SSE against the number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()


# In[7]:


# Apply K-means clustering
k = 4  # Chosen based on the Elbow Method
kmeans = KMeans(n_clusters=k, random_state=0)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Display the first few rows of the RFM dataframe with cluster labels
rfm.head()


# In[8]:


import seaborn as sns

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segmentation based on RFM')
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.show()


# ### Recommendations Based on Segmentation
# 
# 1. **Cluster 0**: 
#    - Characteristics: Low Recency, High Frequency, High Monetary.
#    - Recommendation: These are loyal customers. Offer loyalty programs or exclusive discounts to retain them.
# 
# 2. **Cluster 1**: 
#    - Characteristics: High Recency, Low Frequency, Low Monetary.
#    - Recommendation: These are at-risk customers. Implement re-engagement campaigns to bring them back.
# 
# 3. **Cluster 2**: 
#    - Characteristics: Medium Recency, Medium Frequency, Medium Monetary.
#    - Recommendation: These are average customers. Offer promotions to increase their spending.
# 
# 4. **Cluster 3**: 
#    - Characteristics: Low Recency, Low Frequency, Low Monetary.
#    - Recommendation: These are new customers. Focus on nurturing them with welcome offers and engaging content.
# 
# 
