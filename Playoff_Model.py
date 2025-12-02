
import argparse
import numpy as np
import pandas as pd
import joblib

# Reuse helpers from your regular-season GLM
from GLM_Model import (
    prepare_merged_season_data,
    clean_numeric_column,
    harmonize_team_column,
    build_model_tweedie,
    train_and_evaluate,
)

# ---------------------------------------------------
# Correct paths for YOUR actual files
# ---------------------------------------------------

DEF_PATH = "NFL Defense.csv"
OFF_PATH = "NFL Offense.csv"
MISC_PATH = "NFL Misc.csv"

# This is your cleaned TEAM–SEASON playoff file
PLAYOFF_PATH = "Playoff_Data.csv"

# Target to train on (you can swap this later)
PLAYOFF_TARGET_COL = "playoff_wins"

PLAYOFF_MODEL_OUT = "playoff_success_glm.joblib"
PLAYOFF_FEATURES_OUT = "playoff_features_list.joblib"


# ---------------------------------------------------
# Build team-season training dataset for playoffs
# ---------------------------------------------------

def prepare_playoff_training_data(
    def_path: str,
    off_path: str,
    misc_path: str,
    playoff_path: str,
    target_col: str = PLAYOFF_TARGET_COL,
):
    """
    1. Build merged regular-season dataset ung GLM_Model helpers.
    2. Load your NFL_Playoff_Games.csv TEAM-SEASON file.
    3. Merge them on Year + Team.
    """
    # regular-season stats
    merged, numeric_cols = prepare_merged_season_data(def_path, off_path, misc_path)

    # playoff team-season labels
    playoff = pd.read_csv(playoff_path, dtype=str)
    playoff = playoff.rename(columns=lambda c: c.strip())
    playoff = harmonize_team_column(playoff)

    if "Year" not in playoff.columns:
        raise KeyError("Playoff file must include a 'Year' column.")

    playoff["Year"] = playoff["Year"].astype(str).str.strip()

    if target_col not in playoff.columns:
        raise KeyError(f"Target column '{target_col}' not found in playoff file.")

    playoff[target_col] = clean_numeric_column(playoff[target_col])

    playoff_small = playoff[["Year", "Team", target_col]]

    # merge regular-season stats with playoff outcomes
    df = merged.merge(playoff_small, on=["Year", "Team"], how="inner")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    return df, numeric_cols, target_col


def prepare_playoff_features_and_target(df, numeric_cols, target_col):
    """
    Use all numeric regular-season columns EXCEPT the target.
    """
    features = [c for c in numeric_cols if c != target_col]
    if not features:
        raise ValueError("No features found for playoff model.")

    X = df[features]
    y = df[target_col].astype(float).values
    return X, y, features


# ---------------------------------------------------
# Training Pipeline (same pattern as GLM_Model.py)
# ---------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Train Playoff GLM to predict playoff success")
    parser.add_argument("--def_path", default=DEF_PATH)
    parser.add_argument("--off_path", default=OFF_PATH)
    parser.add_argument("--misc_path", default=MISC_PATH)
    parser.add_argument("--playoff_path", default=PLAYOFF_PATH)
    parser.add_argument("--target_col", default=PLAYOFF_TARGET_COL)
    parser.add_argument("--power", type=float, default=0.0, help="Tweedie power")
    parser.add_argument("--alpha", type=float, default=0.0, help="Regularization strength")
    parser.add_argument("--out", default=PLAYOFF_MODEL_OUT)
    parser.add_argument("--features_out", default=PLAYOFF_FEATURES_OUT)
    parsed = parser.parse_args(args)

    # build dataset
    df, numeric_cols, target_col = prepare_playoff_training_data(
        parsed.def_path,
        parsed.off_path,
        parsed.misc_path,
        parsed.playoff_path,
        parsed.target_col,
    )

    X, y, features = prepare_playoff_features_and_target(df, numeric_cols, target_col)

    # train GLM
    print(f"Training playoff model (target={target_col}, power={parsed.power}, alpha={parsed.alpha})")
    model = build_model_tweedie(power=parsed.power, alpha=parsed.alpha)
    model, cv_scores = train_and_evaluate(X, y, power=parsed.power, alpha=parsed.alpha)

    joblib.dump(model, parsed.out)
    joblib.dump(features, parsed.features_out)

    print(f"Saved model to: {parsed.out}")
    print(f"Saved features to: {parsed.features_out}")
    print("cv_r2_mean =", round(np.mean(cv_scores), 3))
    print("cv_r2_std  =", round(np.std(cv_scores), 3))

    glm = model.named_steps["glm"]
    coefs = pd.Series(glm.coef_, index=features)
    coefs_sorted = coefs.sort_values(key=abs, ascending=False)

    print("\nTop stats correlated with playoff success:")
    print(coefs_sorted.head(20))

    return model, features



if __name__ == "__main__":
    main()

