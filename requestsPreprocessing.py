import pandas as pd

requests_df = pd.read_csv('data/requests.csv')
requests_df['HighProtein'] = requests_df['HighProtein'].map({'Indifferent': 0.5, 'Yes': 1})
requests_df['LowSugar'] = requests_df['LowSugar'].map({'Indifferent': 0.5, '0': 0})
# save to requests_clean.csv
requests_df.to_csv('processedData/requests_clean.csv', index=False)
print(pd.read_csv('processedData/requests_clean.csv').head())
