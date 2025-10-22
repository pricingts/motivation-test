# /home/jupyter/Motivations_test/score_results.py
# Scoring for 36-item Likert questionnaire (A1..A12, F1..F12, P1..P12)
# Usage:
#   python score_results.py \
#     --csv /home/jupyter/Motivations_test/data/respuestas_cuestionario.csv \
#     --out /home/jupyter/Motivations_test/data/resultados_scoring.csv \
#     --summary /home/jupyter/Motivations_test/data/resumen_scoring.csv \
#     --dedupe cedula   # (optional) keep latest per cedula

import argparse
from datetime import datetime
import numpy as np
import pandas as pd

L_MIN, L_MAX = 1, 5
MID = (L_MIN + L_MAX) / 2

SUBSCALES = {
    "Logros": {
        "items": [f"A{i}" for i in range(1,13)],
        "reverse": ["A5","A6","A10","A11","A12"],
    },
    "Afiliación": {
        "items": [f"F{i}" for i in range(1,13)],
        "reverse": ["F4","F5","F6","F9","F10","F12"],
    },
    "Poder": {
        "items": [f"P{i}" for i in range(1,13)],
        "reverse": ["P4","P5","P6","P9","P10","P12"],
    },
}

ALL_ITEMS = SUBSCALES["Logros"]["items"] + SUBSCALES["Afiliación"]["items"] + SUBSCALES["Poder"]["items"]

def reverse_score(s: pd.Series, lmin=L_MIN, lmax=L_MAX) -> pd.Series:
    return (lmin + lmax) - s

def percent_from_mean(mean_score: pd.Series, lmin=L_MIN, lmax=L_MAX) -> pd.Series:
    return ((mean_score - lmin) / (lmax - lmin)) * 100.0

def cronbach_alpha(df_sub: pd.DataFrame) -> float:
    df_sub = df_sub.dropna(axis=0, how="any")
    k = df_sub.shape[1]
    if k <= 1 or df_sub.shape[0] == 0:
        return np.nan
    item_vars = df_sub.var(axis=0, ddof=1)
    total_var = df_sub.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return float((k/(k-1.0)) * (1.0 - item_vars.sum() / total_var))

def label_from_percent(p):
    if pd.isna(p):
        return "N/A"
    if p >= 60: return "Alta"
    if p >= 40: return "Moderada"
    return "Baja"

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # Normalize columns and types
    for col in ["timestamp_utc","nombre","cedula"]:
        if col not in df.columns: df[col] = np.nan
    # Cast items to numeric
    for col in ALL_ITEMS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    return df

def dedupe_keep_latest(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in df.columns: return df
    # Parse timestamp; rows without timestamp get NaT and go last
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df = df.assign(_ts=ts)
    # keep last (max timestamp) per key
    idx = df.sort_values("_ts").groupby(key, dropna=False).tail(1).index
    out = df.loc[idx].drop(columns=["_ts"]).sort_index()
    return out

def score_df(df: pd.DataFrame):
    scored = df.copy()

    # Reverse-score required items
    for scale, spec in SUBSCALES.items():
        for col in spec["reverse"]:
            if col in scored.columns:
                scored[col] = reverse_score(pd.to_numeric(scored[col], errors="coerce"))

    # Per-scale means and percents
    results = pd.DataFrame(index=scored.index)
    reliability = {}
    for scale, spec in SUBSCALES.items():
        sub = scored[spec["items"]]
        mean_scores = sub.mean(axis=1)  # no missing if enforced, robust anyway
        results[f"{scale}_media"] = mean_scores
        results[f"{scale}_porc"] = percent_from_mean(mean_scores)
        reliability[scale] = cronbach_alpha(sub)

    # Dominant profile with ties → "Mixto"
    perc_cols = [c for c in results.columns if c.endswith("_porc")]
    max_vals = results[perc_cols].max(axis=1)
    is_tie = results[perc_cols].eq(max_vals, axis=0).sum(axis=1) > 1
    results["perfil_dominante"] = results[perc_cols].idxmax(axis=1).str.replace("_porc","", regex=False)
    results.loc[is_tie, "perfil_dominante"] = "Mixto"

    # Quality flags
    person_std = scored[ALL_ITEMS].std(axis=1)
    results["respuesta_plana_flag"] = person_std < 0.30
    results["aquiescencia"] = scored[ALL_ITEMS].sub(MID).abs().mean(axis=1)

    # Labels (Alta/Moderada/Baja)
    for scale in SUBSCALES.keys():
        results[f"{scale}_nivel"] = results[f"{scale}_porc"].apply(label_from_percent)

    # Round percents
    for c in perc_cols + ["aquiescencia"]:
        results[c] = results[c].round(2)

    # Bring id columns forward
    id_cols = [c for c in ["timestamp_utc","nombre","cedula"] if c in df.columns]
    out = pd.concat([df[id_cols], results], axis=1)

    return out, reliability

def aggregate_summary(scored: pd.DataFrame) -> pd.DataFrame:
    # Means of percents and distribution of dominant profile
    means = scored[[c for c in scored.columns if c.endswith("_porc")]].mean(numeric_only=True).round(2)
    dist = scored["perfil_dominante"].value_counts(dropna=False)
    summary_rows = []
    for k, v in means.items():
        summary_rows.append({"metric": k, "value": v})
    for k, v in dist.items():
        summary_rows.append({"metric": f"perfil_{k}", "value": int(v)})
    return pd.DataFrame(summary_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with raw answers")
    ap.add_argument("--out", required=True, help="Output CSV with scored results")
    ap.add_argument("--summary", required=False, default=None, help="Optional CSV with aggregate summary")
    ap.add_argument("--dedupe", choices=["none","cedula","nombre"], default="none",
                    help="Keep latest per key (based on timestamp_utc)")
    args = ap.parse_args()

    df = load_csv(args.csv)

    if args.dedupe != "none":
        df = dedupe_keep_latest(df, key=args.dedupe)

    scored, reliability = score_df(df)
    scored.to_csv(args.out, index=False, encoding="utf-8")

    if args.summary:
        agg = aggregate_summary(scored)
        agg.to_csv(args.summary, index=False, encoding="utf-8")

    print("=== Scoring done ===")
    print(f"Saved: {args.out}")
    if args.summary:
        print(f"Saved summary: {args.summary}")
    print("Cronbach's alpha by subscale:", reliability)

if __name__ == "__main__":
    main()
