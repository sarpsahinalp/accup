import re

import pandas as pd
from fractions import Fraction


# Create static arrays for the ingredients
def classify_diet(products):
    meat_keywords = ["beef", "pork", "chicken", "lamb", "veal", "venison", "duck", "turkey",
                     "bacon", "sausage", "ham", "salami", "pepperoni", "bison", "rabbit", "quail",
                     "goose", "partridge", "pheasant", "kangaroo", "elk", "buffalo", "boar", "horse"]

    fish_keywords = ["salmon", "tuna", "cod", "trout", "mackerel", "bass", "catfish", "halibut",
                     "tilapia", "snapper", "sardine", "anchovy", "haddock", "sole", "flounder", "swordfish",
                     "shrimp", "lobster", "crab", "clam", "mussel", "oyster", "scallop", "squid"]

    animal_product_keywords = ["egg", "milk", "butter", "cheese", "yogurt", "cream", "honey", "gelatin", "ghee"]

    veg_check = ["vegan", "vegetarian"]

    for product in products:
        product_lower = product.lower()

        # Check for explicit non-vegetarian ingredients
        if any(keyword in product_lower for keyword in meat_keywords):
            if not (any(keyword1 in product_lower for keyword1 in veg_check)):
                return "Omnivore"
        elif any(keyword in product_lower for keyword in fish_keywords):
            if not (any(keyword1 in product_lower for keyword1 in veg_check)):
                return "Omnivore"

    # If no non-vegetarian ingredients found, check for vegetarian/vegan status
    for product in products:
        product_lower = product.lower()
        if any(keyword in product_lower for keyword in animal_product_keywords):
            return "Vegetarian"

    return "Vegan"


def extract_ingredients(s):
    # Use regular expression to find all occurrences of quoted strings
    matches = re.findall(r'\"\\"(.*?)\\"\"', s)
    # Return the list of matched string
    return matches


def sum_of_fractions(arr):
    total = 0
    for s in arr:
        if '-' in s:  # It's a range
            min_val, max_val = s.split('-')
            # Convert to fractions and calculate the average
            avg_val = (convert_to_fraction(min_val.strip()) + convert_to_fraction(max_val.strip())) / 2
            total += avg_val
        else:  # It's a single value
            total += convert_to_fraction(s)
    return total


def convert_to_fraction(s):
    """Convert a string to a Fraction. Handles mixed numbers."""
    if ' ' in s:  # It's a mixed number
        whole, frac = s.split()
        return Fraction(whole) + Fraction(frac)
    else:  # It's a whole number or fraction
        return Fraction(s)


recipes_df = pd.read_csv('data/recipes.csv')

# Get number of ingredients using quantities
recipes_df['Ingredients'] = recipes_df['RecipeIngredientParts'].apply(extract_ingredients)
recipes_df['RecipeDiet'] = recipes_df['Ingredients'].apply(classify_diet)
# Use one hot encoding to convert categorical data to numerical
recipes_df = pd.get_dummies(recipes_df, columns=['RecipeDiet'], dtype=int)

recipes_df['Quantities'] = recipes_df['RecipeIngredientQuantities'].apply(extract_ingredients)
recipes_df['Quantities'] = recipes_df['Quantities'].apply(sum_of_fractions).astype(float)

# Fill recipe servings with mean
recipes_df['RecipeServings'] = (recipes_df['RecipeServings'].fillna(recipes_df['RecipeServings'].mean()))
recipes_df['RecipeCategory'] = recipes_df['RecipeCategory']
# Use one hot encoding to convert categorical data to numerical
recipes_df = pd.get_dummies(recipes_df, columns=['RecipeCategory'], dtype=int)
recipes_df.drop(['RecipeIngredientParts', 'RecipeIngredientQuantities', 'RecipeYield', 'Ingredients', 'Name'], axis=1,
                inplace=True)

# save to csv
recipes_df.to_csv('processedData/recipes_clean.csv', index=False)
print(pd.read_csv('processedData/recipes_clean.csv').head())