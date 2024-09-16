# Loading the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Loading the dataset
df = pd.read_excel(r"Synthetic Agent Data.xlsx")
df.head()

# Getting the information on data
df.info()

## DATA CLEANING ##
# Checking for missing values
df.isnull().sum()

# Dropping the first column
df = df.iloc[:,1:]
df.head()

# Transposing the dataset and renaming the columns
df_transposed = df.transpose()
df_transposed.columns = [f'Day_{i}' for i in range(df_transposed.shape[1])]
df_transposed.head()

# Checking for outliers using the box plot technique
plt.Figure(figsize=(14,6))
sns.boxplot(data=df_transposed)
plt.title('Outliers Box Plot')
plt.show()

# Removing outliers using the IQR method
Q1 = df_transposed.quantile(0.25)
Q3 = df_transposed.quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df_transposed[~((df_transposed < (Q1 - 1.5 * IQR)) | (df_transposed > (Q3 + 1.5 * IQR))).any(axis=1)]

# Confirm outliers removal by plotting a box plot
plt.Figure(figsize=(14,6))
sns.boxplot(data=df_no_outliers)
plt.title('Outliers Removal Box Plot')
plt.show()

## STATISTICAL ANALYSIS ##
# Calculating Mean, Median, Max, and sd (Individual stats)
AgentStats = pd.DataFrame()  # Create a data-frame to store the statistics
AgentStats['Mean'] = df_no_outliers.mean(axis=1)
AgentStats['Median'] = df_no_outliers.median(axis=1)
AgentStats['Sd'] = df_no_outliers.std(axis=1)
AgentStats['Minimum'] = df_no_outliers.min(axis=1)
AgentStats['Maximum'] = df_no_outliers.max(axis=1)

print(AgentStats)

# Calculating Mean, Median, Max, and sd (Overall stats)
OverallStats = pd.DataFrame()  # Create a data-frame to store the statistics
OverallStats['Mean'] = [df_no_outliers.values.mean()]
OverallStats['Median'] = [np.median(df_no_outliers.values)]
OverallStats['Sd'] = [df_no_outliers.values.std()]
OverallStats['Minimum'] = [df_no_outliers.values.min()]
OverallStats['Maximum'] = [df_no_outliers.values.max()]

print(OverallStats)

# Checking the data distribution using histograms (each agent)
plt.figure(figsize=(15,12))
NoOfPlots = min(df_no_outliers.shape[0], 10) # limits no. of plots to 10 agents

for i in range(NoOfPlots):
    plt.subplot(5, 5, 1+i)
    sns.histplot(df_no_outliers.iloc[i], kde=True)
    plt.title(f'Agent {i+1}')
    
plt.tight_layout()
plt.show()

# Normalize the data using StandardScaler and re-plot the histograms
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_no_outliers)
df_normalized = pd.DataFrame(df_normalized, columns = df_no_outliers.columns)

# Plotting the Normalized dataset
plt.figure(figsize=(15,12))
NoOfPlots = min(df_normalized.shape[0], 10) # limits no. of plots to 10 agents

for i in range(NoOfPlots):
    plt.subplot(5, 5, i+1)
    sns.histplot(df_normalized.iloc[i], kde=True)
    plt.title(f'Agent {i+1} (Normalized)')
    
plt.tight_layout()
plt.show()

## CLUSTERING ANALYSIS ##
# Clustering the dataset
# Standardize the data
scaler = StandardScaler()
ScaledAgentStats = scaler.fit_transform(AgentStats)

# Using the 'Elbow' method to determine the optimal no. of clusters
wcss = []  # list to store the within cluster sum of squares values

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(ScaledAgentStats)
    wcss.append(kmeans.inertia_)
    
# Plot the 'Elbow' graph
plt.plot(range(1,11), wcss, marker = 'o')
plt.title('Elbow Method For Optimal Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Assuming k=3 from the 'Elbow' plot
kmeans = KMeans(n_clusters=3, random_state=0).fit(ScaledAgentStats)
AgentStats['Cluster'] = kmeans.labels_

# Analyzing the statistics of each cluster
ClusterStats = AgentStats.groupby('Cluster').mean()
print(ClusterStats)

# Plotting the Cluster Statistics
plt.Figure(figsize=(14, 8))
sns.scatterplot(x='Mean', y='Sd', hue='Cluster', data=AgentStats, palette='Set1')
plt.title('Cluster Statistics of Field Agents (low, moderate and high performers)')
plt.show()

# Evaluating the quality of the clusters using the Silhouette Score
SilhouetteAvg = silhouette_score(ScaledAgentStats, AgentStats['Cluster'])
print(f'The Silhouette score for the 3 clusters is: {round(SilhouetteAvg, 2)}')

## INCENTIVES STRUCTURE DESIGN ##
# Calculating total no. of tasks performed by each agent over the 83 days
TotalTasksPerAgent = df_no_outliers.sum(axis=1)
TotalTasksPerAgent.head() 

# Creating a visualization (histogram) to show distribution of total tasks completed
plt.Figure(figsize=(12, 6))
plt.hist(TotalTasksPerAgent, bins=20, edgecolor='black')
plt.xlabel('Total Task Completed')
plt.ylabel('Frequency')
plt.title('Distribution Of Total Tasks Completed by Field Agents')
plt.show()

# Initializing and displaying the compensation
BasePay = 175   # Baseline payment in KSH
print(f'Baseline Payment per Task: KSH {BasePay}')

# Average bonus per task in KSH
AvgBonus = 100
print(f'Average Bonus per Task: KSH {AvgBonus}')

# Total Average compensation per task (in KSH)
AvgCompensation = BasePay + AvgBonus

# Ensuring the Total Avg Compensation per Task is limited to a maximum of KSH 300
if AvgCompensation > 300:
    print('WARNING! The Avg Compensation per Task exceeds the required limit of KSH 300')
else:
    print(f'Total Average Compensation Per Task: KSH {AvgCompensation}')
    
# Apply on actual dataset
def CalcTotalCompensation(TasksCompleted):
    BasePay = 175   # Base payment per task in KSH
    
    if TasksCompleted <= 25:
        bonus = 0   # bonus per task in KSH
    elif 25 < TasksCompleted <= 50:
        bonus = 25
    elif 50 < TasksCompleted <= 75:
        bonus = 75
    else:           # TasksCompleted > 75
        bonus = 125 
        
    # Total compensation per task
    TotalCompensation = BasePay + bonus
    return TotalCompensation

# Total tasks completed by the field agents over the 83-day period
TotalTasksPerAgent

# Merge the total tasks completed by each agent and their corresponding compensation
CompensationPerTask = TotalTasksPerAgent.apply(CalcTotalCompensation)
CompensationSummary = pd.DataFrame({
    'Total Tasks Completed': TotalTasksPerAgent,
    'Total Compensation per Task (KSH)': CompensationPerTask
})

# Display summary on first 10 agents
CompensationSummary.head(10)

## FINANCIAL EVALUATION ##
# Calculating no. of field agents in each bonus group
AgentsCount = pd.cut(CompensationSummary['Total Tasks Completed'],
                     bins=[0, 25, 50, 75, float('inf')],
                     labels=('Ksh 0 bonus', 'Ksh 50 bonus', 'Ksh 100 bonus', 'Ksh 125 bonus')).value_counts()
print(AgentsCount)

## SCENARIO 1 ##
# Calculating the no. of agents in Tiers 1 & 2
AgentsTier1_2 = {
    'Tier 1': 9,    # agents that complete 0-25 tasks
    'Tier 2': 460,   # agents that complete 26-50 tasks
}

# Compensation Per Task in Tier 1 and 2
CompensationTiers1_2 = {
    'Tier 1': 175,    # KSH tasks
    'Tier 2': 225,   # KSH per tasks
}

# Calculating the Avg Compensation in Tiers 1 and 2 
TotalAgentsTier1_2 = sum(AgentsTier1_2.values())
TotalCompensationTiers1_2 = sum(AgentsTier1_2[tier] * CompensationTiers1_2[tier]
                                for tier in AgentsTier1_2)
AvgCompensationTier1_2 = TotalCompensationTiers1_2 / TotalAgentsTier1_2
print(f'Total Number of Agents in Tier 1 and Tier 2: {TotalAgentsTier1_2}')
print(f'Total Compensation per Task for Agents in Tier 1 and Tier 2: KSH {TotalCompensationTiers1_2}')
print(f'Average Compensation per Task in Tiers 1 and 2: KSH {round(AvgCompensationTier1_2,2)}')

## SCENARIO 2 ##
# Agents in Tier 3
AgentsTier3 = {
    'Tier 3': 25    # agents that complete 51-75 tasks
}
TotalAgentsTier3 = sum(AgentsTier3.values())

# Compensation per task in Tier 3
CompensationTier3 = {
    'Tier 3': 275
}

# Average Compensation Per Task in Tier 3
TotalCompensationTier3 = sum(CompensationTier3[tier] * AgentsTier3[tier]
                             for tier in AgentsTier3)
AvgCompensationTier3 = TotalCompensationTier3 / TotalAgentsTier3
print(f'Total Number of Agents in Tier 3: {TotalAgentsTier3}')
print(f'Total Compensation per Task for Agents in Tier 3: KSH {TotalCompensationTier3}')
print(f'Average Compensation per Task in Tier 3: KSH {round(AvgCompensationTier3,2)}')

## SCENARIO 3 ##
# Agents in Tier 4
AgentsTier4 = {
    'Tier 4': 704    # agents that complete 51-75 tasks
}
TotalAgentsTier4 = sum(AgentsTier4.values())

# Compensation per task in Tier 4
CompensationTier4 = {
    'Tier 4': 300
}

# Average Compensation Per Task in Tier 3
TotalCompensationTier4 = sum(CompensationTier4[tier] * AgentsTier4[tier]
                             for tier in AgentsTier4)
AvgCompensationTier4 = TotalCompensationTier4 / TotalAgentsTier4
print(f'Total Number of Agents in Tier 4: {TotalAgentsTier4}')
print(f'Total Compensation per Task for Agents in Tier 4: KSH {TotalCompensationTier4}')
print(f'Average Compensation per Task in Tier 4: KSH {round(AvgCompensationTier4,2)}')

# Visualization
AvgCompensation = {
    'Scenario 1': AvgCompensationTier1_2,
    'Scenario 2': AvgCompensationTier3,
    'Scenario 3': AvgCompensationTier4
}

# Bar Graph representation of avg compensation in each Scenario
plt.Figure(figsize=(12, 6))
plt.bar(AvgCompensation.keys(), AvgCompensation.values(), color='slateblue')
plt.title('Average Compensation Per Task Across Scenarios 1,2 & 3')
plt.xlabel('Scenario')
plt.ylabel('Average compensation (KSH)')
plt.show()

# Total Compensation
# No. of agents in each tier
AgentsPerTier = {
    'Tier 1': 9,    # KSH per task
    'Tier 2': 460,
    'Tier 3': 25,
    'Tier 4': 704
}

# Compensation per task in each tier
CompensationPerTier = {
    'Tier 1': 175,    # KSH per task
    'Tier 2': 225,
    'Tier 3': 275,
    'Tier 4': 300
}

# Total compensation cost across all field agents
TotalCompensationCost = sum(AgentsPerTier[tier] * CompensationPerTier[tier]
                            for tier in AgentsPerTier)

print(f'Total Compensation Cost: KSH {TotalCompensationCost}')