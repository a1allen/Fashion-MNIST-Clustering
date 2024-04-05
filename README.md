# Fashion MNIST Clustering

## Overview
As someone who is passionate about fashion, technology, and machine learning, I wanted to investigate these fields together using this [Fashion MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist). This Python script demonstrates the application of K-means clustering on the dataset, helping to draw conclusions about patterns and groupings within various pieces of clothing. 

Creating this script has been fulfilling and enjoyable. I extend my gratitude to the creator of the dataset for providing such valuable data, and for the opportunity to extract insights and draw meaningful conclusions from it.

This script performs the following steps:

1. **Loading Data:** The Fashion MNIST dataset is loaded from CSV files downloaded from Kaggle.
2. **Preprocessing:** Data preprocessing involves splitting the dataset into features and labels, and standardizing the features using `StandardScaler`.
3. **Clustering:** K-means clustering with 10 clusters is applied to the preprocessed data.
4. **Visualization:** Principal Component Analysis (PCA) is used for dimensionality reduction to visualize the clusters in a 2D scatter plot. Each cluster is color-coded, and a legend is provided to indicate the clothing class associated with each cluster.

## How to Use
1. Clone the repository from GitHub.
2. Download the [Fashion MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist) from Kaggle and place the CSV files in the same folder as this script.
3. Run the script in a Python environment, preferably using Spyder IDE for simple visualization.

## Dependencies
- pandas
- scikit-learn
- matplotlib
