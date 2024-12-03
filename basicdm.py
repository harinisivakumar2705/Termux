import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load a free dataset, e.g., a sample dataset available online
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Display first few rows of the dataset
print(data.head())

# Basic Stats
print("\nBasic Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Display a histogram for a column (e.g., 'Age')
plt.hist(data['Age'].dropna(), bins=20, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Correlation Matrix
print("\nCorrelation Matrix:")
print(data.corr())

# Display a correlation heatmap
import seaborn as sns
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
