#we have a csv file listing all the super bowl winners and losers from 2001 to 2025
#We also have a csv file listing the offense, defense, and miscallaneous stats for each team from 2000 to 2024

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict #for cross validation
from sklearn.preprocessing import StandardScaler #for feature scaling
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, confusion_matrix #for model evaluation


nfl_data = pd.read_csv('merged_nfl_data.csv')
sb_data = pd.read_csv('superbowl_teams.csv')

# test the data loading
print(nfl_data.head())
print(nfl_data.shape)
print(sb_data.head())
print(sb_data.shape)

#create training data
max_nfl_year = nfl_data['Year'].max()
sb_train = sb_data[sb_data['NFL Year'] <= max_nfl_year].copy()

id_cols = ["Year", "Team"]
feature_cols = [c for c in nfl_data.columns if c not in id_cols] #create feature columns

rows, labels, groups = [], [], []

for _, row in sb_train.iterrows():
    nfl_year = int(row["NFL Year"]) #we use nfl season start year instead of the actual superbowl year for indexing
    winner, loser = row["Winner"], row["Loser"]

    A = nfl_data[(nfl_data["Year"] == nfl_year) & (nfl_data["Team"] == winner)].iloc[0]
    B = nfl_data[(nfl_data["Year"] == nfl_year) & (nfl_data["Team"] == loser)].iloc[0]

    diff = A[feature_cols].astype(float) - B[feature_cols].astype(float) #create difference features

    # (Winner - Loser) labeled 1
    rows.append(diff)
    labels.append(1)
    groups.append(nfl_year)

    # (Loser - Winner) labeled 0, for balance
    rows.append(-diff)
    labels.append(0)
    groups.append(nfl_year)

X = pd.DataFrame(rows)
y = np.array(labels)
groups = np.array(groups)

#logistic regression
logit_pipe = Pipeline([
    ("scaler", StandardScaler()),  #feature scaling
    ("logit", LogisticRegression(penalty="l2", C=0.01, solver="liblinear")) #we use a low C for regularization
])

# Cross validation
logo = LeaveOneGroupOut()

proba = cross_val_predict(
    logit_pipe, X, y,
    cv=logo, groups=groups,
    method="predict_proba"
)[:, 1]

pred = (proba >= 0.5).astype(int)

print("accuracy:", accuracy_score(y, pred))
print("logloss :", log_loss(y, proba))
print("brier   :", brier_score_loss(y, proba))
print("confusion matrix:\n", confusion_matrix(y, pred))

#Fit final model
logit_pipe.fit(X, y)

#Prediction section
def predict_sb(teamA_stats, teamB_stats):
    diff = (
        teamA_stats[feature_cols].astype(float) -
        teamB_stats[feature_cols].astype(float)
    ).to_frame().T

    return logit_pipe.predict_proba(diff)[0, 1]  # P(Team A wins)


pred_2025 = pd.read_csv("nfl_25_stats.csv")
season_year = 2025  #season year is 1 year before superbowl year

missing = set(feature_cols) - set(pred_2025.columns)
extra   = set(pred_2025.columns) - set(feature_cols) - {"Year","Team"}
print("Missing cols in pred_2025:", missing)
print("Extra cols in pred_2025:", extra)

# Team names for prediction:
teamA = "Buffalo Bills"
teamB = "Los Angeles Rams"

assert teamA in pred_2025["Team"].values, f"{teamA} not found"
assert teamB in pred_2025["Team"].values, f"{teamB} not found"

teamA_stats = pred_2025[(pred_2025["Year"]==season_year) &
                        (pred_2025["Team"]==teamA)].iloc[0]

teamB_stats = pred_2025[(pred_2025["Year"]==season_year) &
                        (pred_2025["Team"]==teamB)].iloc[0]

pA = predict_sb(teamA_stats, teamB_stats)

print(f"P({teamA} wins Super Bowl LX) =", pA)
print(f"P({teamB} wins Super Bowl LX) =", 1-pA, "\n\n")


# Extract scaler and logit model
scaler = logit_pipe.named_steps["scaler"]
logit  = logit_pipe.named_steps["logit"]

# Get coefficients mapped to feature names
coef_series = pd.Series(logit.coef_[0], index=feature_cols)

# Compute raw (Bills - Rams) diff vector
raw_diff = (teamA_stats[feature_cols].astype(float) -
            teamB_stats[feature_cols].astype(float))

# Scale using same scaler as training
scaled_diff = pd.Series(
    scaler.transform(raw_diff.to_frame().T)[0],
    index=feature_cols
)

# Contribution to log-odds = coef * scaled_diff
log_odds_contrib = coef_series * scaled_diff

# Build output table
contrib_table = pd.DataFrame({
    "Feature": feature_cols,
    "Weight (coef)": coef_series.values,
    "Diff (Bills - Rams)": raw_diff.values,
    "Diff (scaled)": scaled_diff.values,
    "Log-odds Contribution": log_odds_contrib.values
})

# SORT BY COEFFICIENT (weight) descending
contrib_table = contrib_table.sort_values(
    "Weight (coef)",
    ascending=False
)

print(contrib_table.to_string(index=False))

