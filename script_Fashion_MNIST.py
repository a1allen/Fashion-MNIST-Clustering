import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

# Load the test and train datasets
#Dataset link: https://www.kaggle.com/datasets/zalando-research/fashionmnist/
#Files are downloaded for convenience
train_data = pd.read_csv('Kaggle_Fashion_MNIST/fashion-mnist_train.csv')
test_data = pd.read_csv('Kaggle_Fashion_MNIST/fashion-mnist_test.csv')

# Get the class labels for the train and test dataset
y_train = train_data.iloc[:, 0].values
y_test = test_data.iloc[:, 0].values

# Get the data outside of the class labels
X_train = train_data.iloc[:, 1:].values
X_test = test_data.iloc[:, 1:].values

# Pre-process data by standardizing it's features 
# Since k-means is sensitive to disruption due to larger-scale features, we use the StandarddizeScalar method
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Perform  on pre-processed data
kmeans = KMeans(n_clusters=10, random_state=42)
y_pred = kmeans.fit_predict(X_train_scaled)

# Help to visualize the data easier by finding the best axis for reconstruction with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Visualize clusters with scatter plot
colours = ["#FF0000", "#FF1493", "#FFA500", "#ffea00", "#3ae83a", "#54ffda", "#24bfff", "#0d45ff", "#8A2BE2", "#EE82EE"]
custom_cmap = ListedColormap(colours)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_pred, cmap=custom_cmap, alpha=0.5)

#Make legend to show which cluster is associated with which colour
legend_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
legend_elements = []

for i in range(len(legend_labels)):
    label = legend_labels[i]
    colour = colours[i]
    marker = plt.scatter([], [], color=colour, label=label, s=100)
    legend_elements.append(marker)

# Display the legend
plt.legend(handles=legend_elements)
plt.title('K-means Clustering of Fashion MNIST Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

