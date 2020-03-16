import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 

from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, plot_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Open dataframe
df = pd.read_csv("data.csv")

# Store labels
y = df["diagnosis"]
# Drop useless columns
df = df.drop(columns=['Unnamed: 32', 'id', 'diagnosis'])

model = LogisticRegression()
model.fit(df, y)
scores = cross_val_score(model, df, y, cv=5)

print(scores, np.mean(scores), np.std(scores))
plot_importance(model)
plt.show()