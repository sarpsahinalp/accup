import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

seed = 2024

# pandas, statsmodels, matplotlib and y_data_profiling rely on numpy's random generator, and thus, we need to set the seed in numpy
np.random.seed(seed)


# Load preprocessed data
recipes_df = pd.read_csv('processedData/recipes_clean.csv')
reviews_df = pd.read_csv('processedData/reviews_clean.csv')
diet_df = pd.read_csv('processedData/diet_clean.csv')
requests_df = pd.read_csv('processedData/requests_clean.csv')

# Merge Requests and Reviews
requests_reviews_df = pd.merge(requests_df, reviews_df, on=['AuthorId', 'RecipeId'], how='inner')

# Merge Requests_Reviews and Recipes
requests_reviews_recipes_df = pd.merge(requests_reviews_df, recipes_df, on='RecipeId', how='inner')

# Merge Requests_Reviews_Recipes and Diet
final_df = pd.merge(requests_reviews_recipes_df, diet_df, on='AuthorId', how='inner')

# Get recipe popularity order by number of likes
recipe_popularity = final_df.groupby(['RecipeId']).sum()['Like'].sort_values(ascending=False)

final_df['RecipePopularity'] = final_df['RecipeId'].map(recipe_popularity)

# Use random forest to predict Like
X = final_df.drop(['Like', 'AuthorId', 'RecipeId'], axis=1)
y = final_df['Like']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rf = GradientBoostingClassifier(n_estimators=500)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Evalueate model using balanced accuracy
print(balanced_accuracy_score(y_test, y_pred))


