import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv')

# Filter the data for 'Satisfied With Life 1' being "Strongly agree", "Agree", "Neither agree nor disagree", or "Disagree"
filtered_df = df[df['Satisfied With Life 1'].isin(["Strongly agree", "Agree","Slightly agree", "Strongly disagree", "Disagree","Slightly disagree"])]

# Drop rows with missing 'Age' values
filtered_df = filtered_df.dropna(subset=['Age'])

# Ensure 'Age' is an integer
filtered_df['Age'] = filtered_df['Age'].astype(int)

# Find the smallest and largest age
smallest_age = filtered_df['Age'].min()
largest_age = filtered_df['Age'].max()

# Create 5 age ranges based on the smallest and largest age
age_bins = [smallest_age, smallest_age + (largest_age - smallest_age) // 5, smallest_age + 2 * (largest_age - smallest_age) // 5, smallest_age + 3 * (largest_age - smallest_age) // 5, smallest_age + 4 * (largest_age - smallest_age) // 5, largest_age + 1]
age_labels = [f'{age_bins[i]}-{age_bins[i+1]-1}' for i in range(len(age_bins)-1)]
filtered_df['Age Range'] = pd.cut(filtered_df['Age'], bins=age_bins, labels=age_labels, right=False)

# Create a new column for satisfaction categories
filtered_df['Satisfaction'] = filtered_df['Satisfied With Life 1'].apply(lambda x: 'Agree' if x in ["Strongly agree", "Agree"] else 'Disagree')

# Plot distribution by age range
age_satisfaction_counts = filtered_df.groupby(['Age Range', 'Satisfaction']).size().unstack().fillna(0)
age_satisfaction_ratios = age_satisfaction_counts.div(age_satisfaction_counts.sum(axis=1), axis=0)

# Plot with white and black bars
fig, ax = plt.subplots(figsize=(10, 6))
age_satisfaction_ratios.plot(kind='bar', stacked=True, ax=ax, color=['white', 'black'], edgecolor='black')
plt.title('Distribution of Satisfaction by Age Range')
plt.xlabel('Age Range')
plt.ylabel('Ratio')
plt.xticks(rotation=45)
plt.legend(title='Satisfaction', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('age.png',dpi=300)
plt.show()

# Plot distribution by gender
gender_satisfaction_counts = filtered_df.groupby(['Gender', 'Satisfaction']).size().unstack().fillna(0)
gender_satisfaction_ratios = gender_satisfaction_counts.div(gender_satisfaction_counts.sum(axis=1), axis=0)

# Plot with white and black bars
fig, ax = plt.subplots(figsize=(10, 6))
gender_satisfaction_ratios.plot(kind='bar', stacked=True, ax=ax, color=['white', 'black'], edgecolor='black')
plt.title('Distribution of Satisfaction by Gender')
plt.xlabel('Gender')
plt.ylabel('Ratio')
plt.xticks(rotation=0)
plt.legend(title='Satisfaction', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('gender.png',dpi=300)
plt.show()

