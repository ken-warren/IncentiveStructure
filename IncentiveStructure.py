import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the data
file_path = 'C:\Users\ADMIN\IncentiveStructure\IncentiveStructure-1\docs\Synthetic Agent Data.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Calculate descriptive statistics
agent_stats = data.describe().T
agent_stats['median'] = data.median()

# Combine data for all agents
combined_data = data.values.flatten()
combined_stats = {
    'mean': np.mean(combined_data),
    'median': np.median(combined_data),
    'max': np.max(combined_data),
    'min': np.min(combined_data),
    'std': np.std(combined_data)
}

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=0)
data['cluster'] = kmeans.fit_predict(scaled_data)

# Evaluate clustering
sil_score = silhouette_score(scaled_data, data['cluster'])
print(f'Silhouette Score: {sil_score}')

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=data['cluster'], palette='viridis')
plt.xlabel('Day 1 Tasks')
plt.ylabel('Day 2 Tasks')
plt.title('Agent Clusters')
plt.show()

# Define the base payment and bonus structure
base_payment = 175
max_average_payment = 300

# Calculate the average number of tasks per agent per week
data['total_tasks'] = data.sum(axis=1)
data['weekly_tasks'] = data['total_tasks'] / (83 / 7)  # Assuming 83 days of data

# Define the bonus tiers
def calculate_bonus(weekly_tasks):
    if weekly_tasks >= 150:
        return 3000
    elif weekly_tasks >= 100:
        return 1500
    elif weekly_tasks >= 50:
        return 500
    else:
        return 0

# Apply the bonus calculation
data['weekly_bonus'] = data['weekly_tasks'].apply(calculate_bonus)

# Calculate the total compensation for each agent
data['total_compensation'] = data['total_tasks'] * base_payment + data['weekly_bonus']
data['average_compensation_per_task'] = data['total_compensation'] / data['total_tasks']

# Ensure the average compensation per task is within the limit
assert data['average_compensation_per_task'].mean() <= max_average_payment, "Average compensation per task exceeds the limit!"

# Save the results to a new Excel file
output_file_path = 'path_to_your_output_file.xlsx'  # Replace with your desired output file path
data.to_excel(output_file_path, index=False)

print("Incentive structure calculated and saved successfully.")