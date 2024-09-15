import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
n=15
# Load the dataset
df = pd.read_csv('Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv')

# Drop 'Survey ID' and 'Qualtiy of Life'
df = df.drop(columns=['Survey ID', 'Qualtiy of Life'])

# Define the target variable
target = df['Quality of Life']
df = df.drop(columns=['Quality of Life'])

# Function to calculate chi-squared and p-values
def chi_squared_test(feature, target):
    contingency_table = pd.crosstab(feature, target)
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    return chi2, p

# Calculate chi-squared and p-values for each feature
results = []
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].astype('category').cat.codes
    df[column] = df[column].fillna(df[column].mode()[0])
    chi2, p = chi_squared_test(df[column], target)
    results.append((column, chi2, p))

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['Feature', 'Chi-Squared', 'P-Value'])

# Print the unadjusted chi-squared values and p-values for comparison
print("\nUnadjusted Results:")
filtered_df = results_df[results_df['P-Value'] < 0.05]
sorted_filtered_df = filtered_df[['Feature', 'Chi-Squared', 'P-Value']].sort_values(by='Chi-Squared', ascending=False)
print(sorted_filtered_df.head(n))
sorted_filtered_df.to_csv('filtered_sorted_results.csv', index=False)
print("\n")

# Adjust p-values using different methods
results_df['Bonferroni Adjusted P-Value'] = multipletests(results_df['P-Value'], method='bonferroni')[1]
results_df['Holm Adjusted P-Value'] = multipletests(results_df['P-Value'], method='holm')[1]
results_df['Benjamini-Hochberg Adjusted P-Value'] = multipletests(results_df['P-Value'], method='fdr_bh')[1]

# Filter features with adjusted p-value < 0.05 and sort by chi-squared
filtered_results_bonferroni = results_df[results_df['Bonferroni Adjusted P-Value'] < 0.05].sort_values(by='Chi-Squared', ascending=False).head(n)
filtered_results_holm = results_df[results_df['Holm Adjusted P-Value'] < 0.05].sort_values(by='Chi-Squared', ascending=False).head(n)
filtered_results_bh = results_df[results_df['Benjamini-Hochberg Adjusted P-Value'] < 0.05].sort_values(by='Chi-Squared', ascending=False).head(n)

# Print the top n features for each adjustment method
print("Bonferroni Adjusted Results:")
print(filtered_results_bonferroni[['Feature', 'Chi-Squared', 'Bonferroni Adjusted P-Value']])
print("\nHolm Adjusted Results:")
print(filtered_results_holm[['Feature', 'Chi-Squared', 'Holm Adjusted P-Value']])
print("\nBenjamini-Hochberg Adjusted Results:")
print(filtered_results_bh[['Feature', 'Chi-Squared', 'Benjamini-Hochberg Adjusted P-Value']])

# Save the results to CSV files
filtered_results_bonferroni.to_csv('top_features_bonferroni.csv', index=False)
filtered_results_holm.to_csv('top_features_holm.csv', index=False)
filtered_results_bh.to_csv('top_features_bh.csv', index=False)

