# Field Agent Performance Analysis and Incentive Structure Design

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Tools](#tools)
- [Data Cleaning](#data-cleaning)
- [Data Analysis](#data-analysis)
- [Results](#results)
- [Conclusion](#conclusion)
- [Recommendations](#recommendations)

## Project Overview
This project involves the analysis of field agents at company X, who capture and collect information about farmers, ranging from KYC data to the polygon/outlines of a farmer's field. The aim is to design an effective incentive structure to motivate the field agents and improve their productivity by rewarding the agents based on their input.

## Problem Statement
As Company X tracks the number of tasks completed by each field agents daily over a 83-days period, it aims to design an incentive structure that motivates agents, particularly those close to the top perfomers who need to reach their full potential. The challlenge is to create a fair and effective incentive plan that motivates all the field agents while maintaining the average compensation per task at a minimum of Ksh 175 and a maximum of Ksh 300.

## Objectives
1. To analyze individual and overall agents performance by calculating mean, median, standard deviation, minimum and maximum of the number of tasks completed.
2. To design an easy to understand and implement incentive  by;
 a) Developing a per-task payment rate that meets the financial criteria.
 b) Proposing a performance bonus to motivate the agents near the top performers.
3. To offer data-driven insights into agent performance and recommend ways to improve agent productivity and satisfaction.

## Tools
The following tools were used;

1. **Python**;
- **Pandas** - For data manipulation
- **Numpy** - For numerical operations and calculations
- **Matplotlib** - For data visualizations and plots
- **Seaborn** - For advanced data visualizations
- **sklearn.cluster(KMeans)** - For performing clustering analysis
- **sklearn.preprocessing(StandardScaler)** - Used to normalize data before clustering
- **sklearn.metrics(silhouette_score)** - For assessing how well the data points have been clustered 

2. **Excel** - For data storage and review

3. **Jupyter Notebook** - IDE for writing and running python codes, displaying visuals and documenting the analysis report.

## Data Source
The dataset file can be found [here](https://github.com/ken-warren/IncentiveStructure/blob/main/docs/Synthetic%20Agent%20Data.xlsx). It consists of 0-83 days entries represented as rows and 1201 columns representing the field agents. 

## Data Cleaning
The following steps were undertaken through jupyter notebook to ensure the data was fit for analysis:
- Detecting and dropping any missing values
- Dropping unwanted columns
- Transposing the dataset for better presentation of the dataset
- Renaming the rows and columns
- Detecting and adjusting the outliers using the IQR method

After cleaning the dataset, the data entries were reduced from 1201 to 1198 rows, which now represented the field agents and the column now represented the field agents.

## Data Analysis
The following Python libraries were used for data analysis:
```python
# Loading the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```
The steps involved in achieving the project's objectives were:
1. **Statistical Analysis**: To calculate mean, median, standard deviation, maximum and minimum values and to also draw visuals and insights into the distribution of the data.
2. **Clustering Analysis**: To group the field agents according to their persona.
3. **Incentive Structure Design**: To motivate and reward the field agents according to their performance.
4. **Financial Evaluation**: To ensure financial viability of the incentive structure which should align with the company's financial goals. 

## Results
After conducting data analysis using Jupyter Notebook, the following results were drawn:

1. **Statistical Analysis**

- The average number of tasks completed across all agents is approximately 4 tasks per day.
- The median number of tasks completed is 1 task, thus on most days the agents completed approximately 1 task.
- The minimum and maximum tasks completed on a single day by any agent was 0 and 17 tasks respectively.
- The overall Sd is approximately 4 tasks, which indicates a wider variability in the number of tasks performed across all agents.



2. **Clustering Analysis**

!(image)[https://github.com/ken-warren/IncentiveStructure/blob/6c62801c4b728715c363d3e7dfef80fb3ed127db/img/ElbowMethod.jpg]
- The 'elbow' is around 3 clusters, it suggests that 3 is the optimal number of clusters for this data. Based on this inference, we will have the 3 clusters as; low, moderate and high performers. 

!(Image)[img\Clusters.jpg]

*Low Performers (Red)*
- These Field Agents have low mean values for tasks performed, approximately below 2 tasks.
- They also have a low standard deviation of approximately 1.
- This suggests that the agents in the red zone complete a very low number of tasks.

*Moderate Performers (Green)*
- These Field Agents have mean values ranging from 2 to 6 for tasks performed.
- They also have a wider standard deviation ranging approximately 1 to 5 indicating varying levels of consistency.
- This suggests that the moderate agents in the green zone may have days of high performance in tasks leading to higher standard deviation or have days of low performance in tasks leading to lower standard deviation.

*High Performers (Blue)*
- These Field Agents have high mean values for tasks performed, approximately above 6 tasks.
- They have a moderate standard deviation ranging approximately 2 to 4.
- This suggests that while the agents in the blue zone are highly productive, there is still some variability in the number of tasks they perform.

To asses the quality of the clustering analysis, the silhouette score was used as shown below;
```python
# Evaluating the quality of the clusters using the Silhouette Score
SilhouetteAvg = silhouette_score(ScaledAgentStats, AgentStats['Cluster'])
print(f'The Silhouette score for the 3 clusters is: {round(SilhouetteAvg, 2)}')
```
The score was high (`0.83`), suggesting the clustering was of good quality.

3. **Incentive Structure Design**

The Python code below shows the way the Incentives Structure was designed step by step.
```python
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
```
To keep the Compensation per task within the financial limit of KSH 300, the bonuses KSH 0, KSH 50, KSH 100 and KSH 125 per task were used. This aimed to motivate the field agents at different performance levels. The bonus groups are explained as follows:

1. **KSH 0 bonus (0-25 tasks)**
- This level only catered for the base payment of KSH 175 per task for field agents who completed few tasks (0-25 task).
- No bonus assigned was aimed at encouraging them to perform more tasks and hence get an additional compensation.

2. **KSH 50 bonus (26-50 tasks)**
- This level catered for the base payment of KSH 175 and an additional bonus of KSH 50, totaling to KSH 225 compensation per task.
- The KSH 25 bonus aims to motivate the agents in this performance level to move to a more productive performance level.

3. **KSH 100 bonus (51-75 tasks)**
- The compensation for agents in this level totals to KSH 275, inclusive of a significant bonus of KSH 100 per task.
- This range targets field agents who are moderately reproductive, motivating them to aim for higher productivity.

4. **KSH 125 bonus (75+ tasks)**
- The compensation in this level totals to the required financial limit of KSH 300 per task, with a maximum bonus of KSH 125 per task.
- This bonus is designed to reward the top performers, encouraging the maximize their output.

4. **Financial Evaluation**

The total compensation cost per task across all agents is approximately KSH 323,150.
!(image)[img\AvgCompensation.jpg]

## Conclusion

The incentive structure designed in this project effectively balances motivating field agents to improve their performance and aim for higher levels, while also maintaining the financial constraints of KSH 175 to KSH 300 per task for compensation.
The following conclusions are drawn from this project:

1. **Scenario Analysis**
- *Scenario 1 (Tiers 1 and 2)*: The model remains cost effective and aligns with the financial limits goal even with most field agents falling in these tiers as the average compensations is approximately KSH 224.
- *Scenario 2 (Tier 3)*: The average compensation per task in this tier is KSH 275, which is below the KSH 300 financial limit. This means that the model is still well designed to manage higher levels of performance without exceeding the required budget.
- *Scenario 3 (Tier 4)*: The average compensation per task at this level is KSH 300. Therefore, the financial budget remains under control even when some agents display top performance with more rewarding bonuses.

2. **Compensation Analysis**
- Those in tier 4 had a bigger representation with 704 agents achieving the KSH 125 bonus reward. This concludes that a big proportion of the field agents are being effectively motivated to reach higher levels of performance which aligns with the company's goal of improved performance across all agents.

3. **Total Compensation Cost**
- The total compensation as estimated in this project is KSH 323,150. This cost reflects the financial viability of the incentive structure and it aligns with the company's financial budget. Therefore this incentive structure has proven to be both effective, affordable and sustainable.

## Recommendations
1. Clearly communicate details of the incentives structure to the agents to ensure openness and motivate them to aim for higher bonuses.
2. Monitor performance of the incentive structure by checking dataset updates, adjust the bonuses and targets as needed.
3. Periodically assess the financial viability of the incentive design and perform cost-benefit analysis.
4. Establish a feedback mechanism where agents can share their thoughts on the incentive structure.
