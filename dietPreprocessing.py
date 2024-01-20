import pandas as pd

diet_df = pd.read_csv('data/diet.csv')
diet_df = diet_df.dropna(subset=['Diet'])
# Use one hot encoding to convert categorical data to numerical
diet_df = pd.get_dummies(diet_df, columns=['Diet'], dtype=int)

# save to diet_clean.csv
diet_df.to_csv('processedData/diet_clean.csv', index=False)
print(pd.read_csv('processedData/diet_clean.csv').head())