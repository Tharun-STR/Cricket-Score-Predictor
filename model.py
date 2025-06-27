import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample training data
data = {
    'runs': [50, 70, 100, 120, 150, 200, 180, 90],
    'wickets': [1, 2, 3, 2, 4, 5, 3, 2],
    'overs': [5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 16.0, 8.0],
    'runs_last_5': [30, 35, 50, 45, 60, 70, 65, 40],
    'batting_team': ['India', 'India', 'Australia', 'Australia', 'England', 'England', 'Pakistan', 'Pakistan'],
    'bowling_team': ['Pakistan', 'England', 'India', 'Pakistan', 'India', 'Pakistan', 'India', 'England'],
    'final_score': [160, 180, 220, 200, 250, 280, 260, 170]
}

df = pd.DataFrame(data)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['batting_team', 'bowling_team'])

X = df_encoded.drop('final_score', axis=1)
y = df_encoded['final_score']

model = LinearRegression()
model.fit(X, y)

# Save model and column names
joblib.dump(model, 'score_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
