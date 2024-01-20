import pandas as pd

reviews_df = pd.read_csv('data/reviews.csv')

# fill rating with mean
reviews_df['Rating'] = reviews_df['Rating'].fillna(reviews_df['Rating'].mean())
reviews_df = reviews_df.dropna(subset=['Like'])
reviews_df = reviews_df.drop(['TestSetId'], axis=1)
reviews_df['Like'] = reviews_df['Like'].astype(int)

# save to reviews_clean.csv
reviews_df.to_csv('processedData/reviews_clean.csv', index=False)
print(pd.read_csv('processedData/reviews_clean.csv').head())