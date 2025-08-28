#!/usr/bin/env python3
"""
OpenPhil Grants Toolkit
-----------------------
Run robust analysis on a grants CSV (like OpenPhilGrants.csv).

Usage:
    python OpenPhil_Grants_Toolkit.py /path/to/your.csv

Outputs (next to the CSV by default, or under /mnt/data/openphil_grants_report when run in ChatGPT):
- Excel workbook with multiple sheets
- CSV extracts for each sheet
- PNG charts
- README_Analysis.md
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r'\s+', ' ', c.strip().lower()) for c in df.columns]
    return df

def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    canon = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in df.columns}
    for name in candidates:
        key = re.sub(r'[^a-z0-9]', '', name.lower())
        if key in canon:
            return canon[key]
        for k, original in canon.items():
            if k.startswith(key) or key in k:
                return original
    return None

def parse_amount(x) -> Optional[float]:
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    s = s.replace(',', '')
    s = re.sub(r'[$€£]', '', s)
    if '-' in s:
        parts = [p for p in s.split('-') if p.strip()]
        try:
            return float(parts[-1])
        except:
            pass
    s = re.sub(r'[A-Za-z]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    try:
        return float(s)
    except:
        return None

def parse_date(x):
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x
    s = str(x).strip()
    s = s.replace('.', '/').replace('-', '/')
    for fmt in ("%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m", "%Y", "%d-%b-%Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt, errors='raise')
        except:
            continue
    try:
        return pd.to_datetime(s, errors='coerce')
    except:
        return None

def top_n_amounts(df: pd.DataFrame, by_col: str, amount_col: str, n: int = 20) -> pd.DataFrame:
    g = df.groupby(by_col, dropna=False)[amount_col].sum().sort_values(ascending=False).head(n)
    return g.reset_index().rename(columns={amount_col: "total_amount_usd"})

def hhi_from_shares(values: pd.Series) -> float:
    total = values.sum()
    if total <= 0:
        return float("nan")
    shares = (values / total)
    return float((shares ** 2).sum())

def run_analysis(input_path: str, out_dir: Optional[str] = None) -> Dict[str, str]:
    input_path = Path(input_path)
    if out_dir is None:
        out_dir = input_path.parent / "openphil_grants_report"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, low_memory=False, encoding="utf-8")
    df = standardize_columns(raw)

    amount_col = pick_column(df, ["amount", "amount (usd)", "amount awarded", "grant amount", "usd", "award (usd)", "amount_usd"])
    date_col = pick_column(df, ["date", "decision date", "award date", "grant date", "start date"])
    recipient_col = pick_column(df, ["recipient", "grantee", "organization", "org", "beneficiary"])
    program_col = pick_column(df, ["program", "focus area", "cause area", "initiative", "subprogram", "area", "theme"])
    title_col = pick_column(df, ["title", "project title", "grant title"])
    purpose_col = pick_column(df, ["purpose", "description", "notes", "summary", "details"])
    country_col = pick_column(df, ["country", "location country", "recipient country"])
    city_col = pick_column(df, ["city", "location city"])

    df["amount_usd"] = df[amount_col].apply(parse_amount) if amount_col else np.nan
    df["decision_date"] = df[date_col].apply(parse_date) if date_col else pd.NaT
    df["year"] = df["decision_date"].dt.year if date_col else np.nan
    df["recipient"] = df[recipient_col].astype(str).str.strip() if recipient_col else None
    df["program_area"] = df[program_col].astype(str).str.strip() if program_col else None
    df["title_norm"] = df[title_col].astype(str).str.strip() if title_col else ""
    df["purpose_norm"] = df[purpose_col].astype(str).str.strip() if purpose_col else ""

    if country_col:
        df["country"] = df[country_col].astype(str).str.strip()
    else:
        df["country"] = np.nan

    if city_col:
        df["city"] = df[city_col].astype(str).str.strip()
    else:
        df["city"] = np.nan

    def is_ai_related(row) -> bool:
        hay = " ".join([str(row.get("program_area", "")),
                        str(row.get("title_norm", "")),
                        str(row.get("purpose_norm", ""))]).lower()
        return any(k in hay for k in ["ai", "artificial intelligence", "alignment", "governance", "ml", "machine learning", "safety"])

    df["is_ai_related"] = df.apply(is_ai_related, axis=1)

    quality = {
        "row_count": len(df),
        "amount_missing_pct": float(df["amount_usd"].isna().mean() * 100),
        "date_missing_pct": float(df["decision_date"].isna().mean() * 100),
        "recipient_missing_pct": float(df["recipient"].isna().mean() * 100) if recipient_col else 100.0,
        "program_area_missing_pct": float(df["program_area"].isna().mean() * 100) if program_col else 100.0,
    }
    quality_df = pd.DataFrame([quality])

    valid_amounts = df["amount_usd"].dropna()
    global_stats = {
        "grants_total": int(df.shape[0]),
        "grants_with_amount": int(valid_amounts.shape[0]),
        "total_amount_usd": float(valid_amounts.sum()),
        "median_grant_usd": float(valid_amounts.median()) if not valid_amounts.empty else np.nan,
        "mean_grant_usd": float(valid_amounts.mean()) if not valid_amounts.empty else np.nan,
        "p90_grant_usd": float(valid_amounts.quantile(0.9)) if not valid_amounts.empty else np.nan,
        "p10_grant_usd": float(valid_amounts.quantile(0.1)) if not valid_amounts.empty else np.nan,
    }
    global_df = pd.DataFrame([global_stats])

    by_year = df.groupby("year", dropna=True).agg(
        grants=("amount_usd", "count"),
        total_amount_usd=("amount_usd", "sum"),
        median_amount_usd=("amount_usd", "median")
    ).reset_index().sort_values("year")

    def calc_cagr(series: pd.Series):
        series = series.dropna()
        if series.shape[0] < 2:
            return None
        try:
            start = series.iloc[0]
            end = series.iloc[-1]
            years = series.shape[0] - 1
            if start and start > 0 and years > 0:
                return (end / start) ** (1 / years) - 1
        except:
            return None
        return None

    cagr_amount = calc_cagr(by_year["total_amount_usd"]) if not by_year.empty else None

    if recipient_col:
        top_recipients_amt = top_n_amounts(df.dropna(subset=["amount_usd"]) , "recipient", "amount_usd", 25)
        top_recipients_cnt = df["recipient"].value_counts().head(25).rename_axis("recipient").reset_index(name="grant_count")
    else:
        top_recipients_amt = pd.DataFrame()
        top_recipients_cnt = pd.DataFrame()

    if program_col:
        top_programs_amt = top_n_amounts(df.dropna(subset=["amount_usd"]), "program_area", "amount_usd", 25)
        top_programs_cnt = df["program_area"].value_counts().head(25).rename_axis("program_area").reset_index(name="grant_count")
    else:
        top_programs_amt = pd.DataFrame()
        top_programs_cnt = pd.DataFrame()

    if "country" in df.columns:
        geo_country_amt = top_n_amounts(df.dropna(subset=["amount_usd"]), "country", "amount_usd", 25)
        geo_country_cnt = df["country"].value_counts().head(25).rename_axis("country").reset_index(name="grant_count")
    else:
        geo_country_amt = pd.DataFrame()
        geo_country_cnt = pd.DataFrame()

    ai_df = df[df["is_ai_related"]].copy()
    ai_stats = {}
    by_year_ai = pd.DataFrame()
    top_ai_recipients_amt = pd.DataFrame()
    top_ai_programs_amt = pd.DataFrame()

    if not ai_df.empty:
        ai_valid = ai_df["amount_usd"].dropna()
        ai_stats = {
            "ai_grants_total": int(ai_df.shape[0]),
            "ai_grants_with_amount": int(ai_valid.shape[0]),
            "ai_total_amount_usd": float(ai_valid.sum()),
            "ai_median_grant_usd": float(ai_valid.median()) if not ai_valid.empty else np.nan,
            "ai_mean_grant_usd": float(ai_valid.mean()) if not ai_valid.empty else np.nan,
        }
        by_year_ai = ai_df.groupby("year", dropna=True).agg(
            grants=("amount_usd", "count"),
            total_amount_usd=("amount_usd", "sum"),
            median_amount_usd=("amount_usd", "median")
        ).reset_index().sort_values("year")
        if recipient_col:
            top_ai_recipients_amt = top_n_amounts(ai_df.dropna(subset=["amount_usd"]), "recipient", "amount_usd", 25)
        if program_col:
            top_ai_programs_amt = top_n_amounts(ai_df.dropna(subset=["amount_usd"]), "program_area", "amount_usd", 25)

    ai_stats_df = pd.DataFrame([ai_stats]) if ai_stats else pd.DataFrame()

    hhi_recipient = hhi_from_shares(df.groupby("recipient")["amount_usd"].sum()) if recipient_col else float("nan")
    hhi_program = hhi_from_shares(df.groupby("program_area")["amount_usd"].sum()) if program_col else float("nan")
    concentration = pd.DataFrame([{"hhi_recipient": hhi_recipient, "hhi_program_area": hhi_program}])

    # Repeat funding
    repeat_metrics = []
    if recipient_col and date_col:
        rec_groups = df.sort_values("decision_date").groupby("recipient", dropna=True)
        for rec, g in rec_groups:
            g = g.dropna(subset=["decision_date"])
            if g.shape[0] == 0:
                continue
            years = g["decision_date"].dt.year.dropna()
            if years.empty:
                continue
            first_year = int(years.min())
            last_year = int(years.max())
            total_amt = float(g["amount_usd"].sum(skipna=True))
            count = int(g.shape[0])
            y_sorted = years.sort_values().tolist()
            if len(y_sorted) > 1:
                intervals = [y_sorted[i+1]-y_sorted[i] for i in range(len(y_sorted)-1)]
                avg_interval = float(np.mean(intervals))
            else:
                avg_interval = float("nan")
            repeat_metrics.append({
                "recipient": rec,
                "grant_count": count,
                "total_amount_usd": total_amt,
                "first_year": first_year,
                "last_year": last_year,
                "avg_interval_years": avg_interval
            })
    repeat_df = pd.DataFrame(repeat_metrics).sort_values(["grant_count", "total_amount_usd"], ascending=[False, False]).head(50)

    # Lightweight text keywords
    def tokenize(text: str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s-]', ' ', text)
        tokens = re.split(r'\s+', text)
        return [t for t in tokens if t and not t.isdigit()]
    stopwords = set("""a an and are as at be by for from has have in into is it its of on or that the to we with this these those their there our your they them his her him you us about towards over under within across between among including include will may can could should would grant research policy program governance safety alignment machine learning ai systems robustness fairness interpretability compute hardware evaluation auditing standards risk regulation law ethics academic fellowship scholarship capacity building lab center institute philanthropic nonprofit advocacy lobbying international global""".split())
    text_source = (df["title_norm"].fillna('') + ' ' + df["purpose_norm"].fillna('')).str.strip()
    tokens = []
    for t in text_source.head(200000):
        tokens.extend([tok for tok in tokenize(t) if tok not in stopwords and len(tok) >= 3])
    from collections import Counter
    word_counts = Counter(tokens)
    top_keywords = pd.DataFrame(word_counts.most_common(50), columns=["term", "count"])

    # Charts
    if not by_year.empty:
        plt.figure(figsize=(10,5))
        plt.plot(by_year["year"], by_year["total_amount_usd"])
        plt.title("Total Amount Awarded by Year (USD)")
        plt.xlabel("Year")
        plt.ylabel("Total Amount (USD)")
        plt.tight_layout()
        plt.savefig(out_dir / "total_amount_by_year.png")
        plt.close()

        plt.figure(figsize=(10,5))
        plt.bar(by_year["year"].astype(int).astype(str), by_year["grants"])
        plt.title("Number of Grants by Year")
        plt.xlabel("Year")
        plt.ylabel("Grant Count")
        plt.tight_layout()
        plt.savefig(out_dir / "grant_count_by_year.png")
        plt.close()

    valid_amounts = df["amount_usd"].dropna()
    if not valid_amounts.empty:
        plt.figure(figsize=(10,5))
        plt.hist(valid_amounts, bins=50)
        plt.title("Grant Amount Distribution (USD)")
        plt.xlabel("Amount (USD)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / "grant_amount_hist.png")
        plt.close()

    if program_col:
        top_programs_amt = top_n_amounts(df.dropna(subset=["amount_usd"]), "program_area", "amount_usd", 25)
        if not top_programs_amt.empty:
            plt.figure(figsize=(10,5))
            plt.bar(top_programs_amt["program_area"].astype(str), top_programs_amt["total_amount_usd"])
            plt.title("Top Program Areas by Total Amount (Top 25)")
            plt.xlabel("Program Area")
            plt.ylabel("Total Amount (USD)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(out_dir / "top_programs_by_amount.png")
            plt.close()

    if recipient_col:
        if not top_recipients_amt.empty:
            plt.figure(figsize=(10,5))
            plt.bar(top_recipients_amt["recipient"].astype(str), top_recipients_amt["total_amount_usd"])
            plt.title("Top Recipients by Total Amount (Top 25)")
            plt.xlabel("Recipient")
            plt.ylabel("Total Amount (USD)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(out_dir / "top_recipients_by_amount.png")
            plt.close()

    sheets: Dict[str, pd.DataFrame] = {}
    sheets["00_Global_Stats"] = pd.DataFrame([global_stats])
    sheets["01_Data_Quality"] = quality_df
    if not by_year.empty:
        sheets["02_By_Year"] = by_year
        sheets["03_CAGR"] = pd.DataFrame([{"cagr_total_amount": cagr_amount}])
    if recipient_col:
        sheets["10_Top_Recipients_Amount"] = top_recipients_amt
        sheets["11_Top_Recipients_Count"] = top_recipients_cnt
    if program_col:
        sheets["20_Top_Programs_Amount"] = top_programs_amt
        sheets["21_Top_Programs_Count"] = top_programs_cnt
    if "country" in df.columns:
        sheets["30_Country_Amount"] = geo_country_amt
        sheets["31_Country_Count"] = geo_country_cnt
    if not repeat_df.empty:
        sheets["40_Repeat_Funding"] = repeat_df
    if not ai_stats_df.empty:
        sheets["50_AI_Stats"] = ai_stats_df
        if not by_year_ai.empty:
            sheets["51_AI_By_Year"] = by_year_ai
        if not top_ai_recipients_amt.empty:
            sheets["52_AI_Top_Recipients"] = top_ai_recipients_amt
        if not top_ai_programs_amt.empty:
            sheets["53_AI_Top_Programs"] = top_ai_programs_amt
    if not top_keywords.empty:
        sheets["60_Top_Keywords"] = top_keywords

    excel_path = out_dir / "OpenPhil_Grants_Analysis.xlsx"
    excel_saved = True
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for name, data in sheets.items():
                sheet_name = name[:31]
                data.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        excel_saved = False

    # CSV extracts
    for name, data in sheets.items():
        safe_name = re.sub(r'[^A-Za-z0-9_\-]+', '_', name)
        data.to_csv(out_dir / f"{safe_name}.csv", index=False)

    # README
    with open(out_dir / "README_Analysis.md", "w", encoding="utf-8") as f:
        f.write("# Open Philanthropy Grants – Automated Analysis\n")
        f.write(f"- Source file: `{input_path}`\n")
        f.write(f"- Rows: {len(df)}\n\n")
        f.write("## What’s inside\n")
        for name in sheets.keys():
            f.write(f"- {name}\n")

    return {
        "excel_path": str(excel_path),
        "excel_saved": str(excel_saved),
        "out_dir": str(out_dir),
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python OpenPhil_Grants_Toolkit.py /path/to/your.csv")
        sys.exit(1)
    res = run_analysis(sys.argv[1])
    print(res)
