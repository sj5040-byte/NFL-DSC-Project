import argparse
import numpy as np
import pandas as pd
import joblib

from GLM_Model import prepare_merged_season_data, clean_numeric_column, harmonize_team_column

# Paths

PREDICTIONS_PATH = "Predictions.csv" 
DEF_PATH = "NFL Defense.csv"
OFF_PATH = "NFL Offense.csv"
MISC_PATH = "NFL Misc.csv"

SEASON_OFF_MISC_PATH = "2025 Offense_Misc.csv"
SEASON_DEF_PATH = "2025 Defense.csv"

PLAYOFF_MODEL_PATH = "playoff_success_glm.joblib"
PLAYOFF_FEATURES_PATH = "playoff_features_list.joblib"

TARGET_YEAR = "2025"

TEAM_TO_CONF_DIV = {
    # AFC East
    "Buffalo Bills": ("AFC", "AFC East"),
    "Miami Dolphins": ("AFC", "AFC East"),
    "New England Patriots": ("AFC", "AFC East"),
    "New York Jets": ("AFC", "AFC East"),

    # AFC North
    "Baltimore Ravens": ("AFC", "AFC North"),
    "Cincinnati Bengals": ("AFC", "AFC North"),
    "Cleveland Browns": ("AFC", "AFC North"),
    "Pittsburgh Steelers": ("AFC", "AFC North"),

    # AFC South
    "Houston Texans": ("AFC", "AFC South"),
    "Indianapolis Colts": ("AFC", "AFC South"),
    "Jacksonville Jaguars": ("AFC", "AFC South"),
    "Tennessee Titans": ("AFC", "AFC South"),

    # AFC West
    "Denver Broncos": ("AFC", "AFC West"),
    "Kansas City Chiefs": ("AFC", "AFC West"),
    "Las Vegas Raiders": ("AFC", "AFC West"),
    "Los Angeles Chargers": ("AFC", "AFC West"),

    # NFC East
    "Dallas Cowboys": ("NFC", "NFC East"),
    "New York Giants": ("NFC", "NFC East"),
    "Philadelphia Eagles": ("NFC", "NFC East"),
    "Washington Commanders": ("NFC", "NFC East"),

    # NFC North
    "Chicago Bears": ("NFC", "NFC North"),
    "Detroit Lions": ("NFC", "NFC North"),
    "Green Bay Packers": ("NFC", "NFC North"),
    "Minnesota Vikings": ("NFC", "NFC North"),

    # NFC South
    "Atlanta Falcons": ("NFC", "NFC South"),
    "Carolina Panthers": ("NFC", "NFC South"),
    "New Orleans Saints": ("NFC", "NFC South"),
    "Tampa Bay Buccaneers": ("NFC", "NFC South"),

    # NFC West
    "Arizona Cardinals": ("NFC", "NFC West"),
    "Los Angeles Rams": ("NFC", "NFC West"),
    "San Francisco 49ers": ("NFC", "NFC West"),
    "Seattle Seahawks": ("NFC", "NFC West"),
}

def add_conf_div(df):
    confs = []
    divs = []
    for team in df["Team"]:
        info = TEAM_TO_CONF_DIV.get(team)
        if info is None:
            confs.append(None)
            divs.append(None)
        else:
            confs.append(info[0])
            divs.append(info[1])
    df["Conference"] = confs
    df["Division"] = divs
    return df


def load_predictions(path):
    preds = pd.read_csv(path)
    preds.columns = preds.columns.str.strip()
    return preds

def seed_conference(df_conf):
    
    df_sorted = df_conf.sort_values(
        ["Division", "pred_wins", "pred_final_win_pct", "expected_playoff_wins"],
        ascending=[True, False, False, False]
    )
    division_winners = (
        df_sorted
        .groupby("Division", as_index=False)
        .head(1)
        .copy()
    )
    mask_winners = df_conf["Team"].isin(division_winners["Team"])
    wild_pool = df_conf[~mask_winners].copy()

    wildcards = wild_pool.sort_values(
        ["pred_wins", "pred_final_win_pct", "expected_playoff_wins"],
        ascending=[False, False, False]
    ).head(3)

    winners_ranked = division_winners.sort_values(
        ["pred_wins", "pred_final_win_pct", "expected_playoff_wins"],
        ascending=[False, False, False]
    ).copy()
    winners_ranked["Seed"] = range(1, 1 + len(winners_ranked))

    wildcards_ranked = wildcards.sort_values(
        ["pred_wins", "pred_final_win_pct", "expected_playoff_wins"],
        ascending=[False, False, False]
    ).copy()
    wildcards_ranked["Seed"] = range(5, 5 + len(wildcards_ranked))

    seeds = pd.concat([winners_ranked, wildcards_ranked], ignore_index=True)
    seeds = seeds.sort_values("Seed").reset_index(drop=True)
    return seeds


def pick_winner(team_a, team_b):
    
    for col in ["expected_playoff_wins", "pred_wins", "pred_final_win_pct"]:
        if team_a[col] > team_b[col]:
            return team_a
        if team_b[col] > team_a[col]:
            return team_b
    # total coin flip fallback: choose alphabetically
    return team_a if team_a["Team"] < team_b["Team"] else team_b


def simulate_conference_bracket(seeds_df, conf_name):
    
    if "Seed" not in seeds_df.columns:
        raise ValueError(
            f"{conf_name} seeds_df missing 'Seed' column. "
            f"Columns are: {list(seeds_df.columns)}"
        )

    seeds_df = seeds_df.copy()
    seeds_df["Seed"] = seeds_df["Seed"].astype(int)
    seeds_df = seeds_df.sort_values("Seed").reset_index(drop=True)

    seed_map = {int(row["Seed"]): row for _, row in seeds_df.iterrows()}

    def get(seed):
        return seed_map[int(seed)]

    s1 = get(1)
    s2 = get(2)
    s3 = get(3)
    s4 = get(4)
    s5 = get(5)
    s6 = get(6)
    s7 = get(7)

    wc_pairs = [
        (2, 7, s2, s7),
        (3, 6, s3, s6),
        (4, 5, s4, s5),
    ]

    winners = [] 

    print(f"\n=== {conf_name} Wild Card Round ===")
    for high_seed, low_seed, team_high, team_low in wc_pairs:
        winner = pick_winner(team_high, team_low)
        
        if winner["Team"] == team_high["Team"]:
            winner_seed = high_seed
        else:
            winner_seed = low_seed

        print(
            f"{high_seed} {team_high['Team']} vs "
            f"{low_seed} {team_low['Team']} -> {winner['Team']}"
        )
        winners.append((winner, winner_seed))

    # Divisional round:

    remaining = sorted(seed for (_, seed) in winners)

    lowest = remaining[-1]    
    highest_two = remaining[:-1]

    seed_to_team = {seed: team for (team, seed) in winners}

    low_team = seed_to_team[lowest]
    high1_team = seed_to_team[highest_two[0]]
    high2_team = seed_to_team[highest_two[1]]

    div_1_low_winner = pick_winner(s1, low_team)
    div_other_winner = pick_winner(high1_team, high2_team)

    print(f"\n=== {conf_name} Divisional Round ===")
    print(
        f"1 {s1['Team']} vs {lowest} {low_team['Team']} "
        f"-> {div_1_low_winner['Team']}"
    )
    print(
        f"{highest_two[0]} {high1_team['Team']} vs "
        f"{highest_two[1]} {high2_team['Team']} "
        f"-> {div_other_winner['Team']}"
    )

    # Conference Championship
    conf_champ = pick_winner(div_1_low_winner, div_other_winner)

    print(f"\n=== {conf_name} Championship Game ===")
    print(
        f"{div_1_low_winner['Team']} vs {div_other_winner['Team']} "
        f"-> {conf_champ['Team']}"
    )

    return conf_champ





def load_2025_snapshot(off_misc_path=SEASON_OFF_MISC_PATH,
                       def_path=SEASON_DEF_PATH):
   
    off_misc = pd.read_csv(off_misc_path, dtype=str)
    defense = pd.read_csv(def_path, dtype=str)

    off_misc = harmonize_team_column(off_misc)
    defense = harmonize_team_column(defense)

    if "Year" in off_misc.columns and "Year" in defense.columns:
        off_misc["Year"] = off_misc["Year"].astype(str).str.strip()
        defense["Year"] = defense["Year"].astype(str).str.strip()
        snapshot = off_misc.merge(defense, on=["Year", "Team"], how="inner")
    else:
        snapshot = off_misc.merge(defense, on="Team", how="inner")

    for col in snapshot.columns:
        if col not in ["Year", "Team"]:
            snapshot[col] = clean_numeric_column(snapshot[col])

    print("2025 snapshot shape:", snapshot.shape)
    return snapshot


def compute_expected_playoff_wins(
    model_path=PLAYOFF_MODEL_PATH,
    features_path=PLAYOFF_FEATURES_PATH,
    off_misc_path=SEASON_OFF_MISC_PATH,
    def_path=SEASON_DEF_PATH,
):
    
    snapshot = load_2025_snapshot(off_misc_path, def_path)

    model = joblib.load(model_path)
    features = joblib.load(features_path)

    for col in features:
        if col not in snapshot.columns:
            snapshot[col] = np.nan

    X = snapshot[features]
    expected = model.predict(X)

    out = snapshot[["Team"]].copy()
    if "Year" in snapshot.columns:
        out["Year"] = snapshot["Year"]
    else:
        out["Year"] = TARGET_YEAR
    out["expected_playoff_wins"] = expected
    return out

def main(args=None):
    parser = argparse.ArgumentParser(description="2025 Playoff Bracket & Predictions")
    parser.add_argument("--predictions_path", default=PREDICTIONS_PATH)
    parser.add_argument("--def_path", default=DEF_PATH)
    parser.add_argument("--off_path", default=OFF_PATH)
    parser.add_argument("--misc_path", default=MISC_PATH)
    parser.add_argument("--playoff_model", default=PLAYOFF_MODEL_PATH)
    parser.add_argument("--playoff_features", default=PLAYOFF_FEATURES_PATH)
    parser.add_argument("--year", default=TARGET_YEAR)
    parsed = parser.parse_args(args)

    preds = load_predictions(parsed.predictions_path)
    preds = add_conf_div(preds)

    playoff_strength = compute_expected_playoff_wins(
        model_path=parsed.playoff_model,
        features_path=parsed.playoff_features,
        off_misc_path=SEASON_OFF_MISC_PATH,
        def_path=SEASON_DEF_PATH,
    )

    full = preds.merge(playoff_strength[["Team", "expected_playoff_wins"]],
                       on="Team", how="left")

    afc = full[full["Conference"] == "AFC"].copy()
    nfc = full[full["Conference"] == "NFC"].copy()

    afc_seeds = seed_conference(afc)
    nfc_seeds = seed_conference(nfc)

    print("\n======= AFC Seeds =======")
    print(afc_seeds[["Seed", "Team", "Division", "pred_wins",
                     "pred_final_win_pct", "expected_playoff_wins"]])

    print("\n======= NFC Seeds =======")
    print(nfc_seeds[["Seed", "Team", "Division", "pred_wins",
                     "pred_final_win_pct", "expected_playoff_wins"]])

    afc_champ = simulate_conference_bracket(afc_seeds, "AFC")
    nfc_champ = simulate_conference_bracket(nfc_seeds, "NFC")


if __name__ == "__main__":
    main()