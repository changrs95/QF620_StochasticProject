#!/usr/bin/env python3
"""
Batch implied volatility calculator for SPX options.

- Iterates over both RTH and ETH folders
- For each trading day, finds:
    YYYY-MM-DD_assets.csv
    YYYY-MM-DD_call_bid_YYYYMMDD.csv
    YYYY-MM-DD_call_ask_YYYYMMDD.csv
    YYYY-MM-DD_put_bid_YYYYMMDD.csv
    YYYY-MM-DD_put_ask_YYYYMMDD.csv
- Extracts expiry as the last 8 digits before .csv (e.g. 20250416)
- Computes implied vol for every row and saves to CSV.

Adjust BASE_DIRS, OUTPUT_DIR, and r/q as needed.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime

# ---------------- CONFIG ----------------

BASE_DIRS = ["RTH", "ETH"]      # folders containing the data
OUTPUT_DIR = "IV_OUTPUT"        # where to save IV csv files

os.makedirs(OUTPUT_DIR, exist_ok=True)

# risk-free rate and dividend yield (annualised)
R = 0.02
Q = 0.0

# ----------------------------------------
# Black–Scholes and IV calculation
# ----------------------------------------


def bs_price(S, K, T, r, sigma, option_type="C", q=0.0):
    """Black–Scholes price for a European call or put."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(0.0, (S - K) if option_type == "C" else (K - S))
        return intrinsic

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, option_type="C", q=0.0, sigma_bounds=(1e-4, 5.0)):
    """
    Solve for implied volatility using a 1D root-finder.

    Returns NaN if no valid solution is found.
    """
    if price is None or np.isnan(price):
        return np.nan
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    intrinsic = max(0.0, (S - K) if option_type == "C" else (K - S))
    if price < intrinsic:
        # Observed price below intrinsic value is inconsistent
        return np.nan

    def f(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - price

    lo, hi = sigma_bounds
    try:
        return brentq(f, lo, hi)
    except Exception:
        return np.nan


# ----------------------------------------
# Helpers to load and reshape data
# ----------------------------------------

def melt_option_wide(df, value_name):
    """
    Convert wide option data (timestamp + strike columns) into long format.

    Input columns: timestamp | 4600 | 4650 | 4700 | ...
    Output: timestamp, strike, <value_name>
    """
    strike_cols = [c for c in df.columns if c != "timestamp"]
    long_df = df.melt(
        id_vars="timestamp",
        value_vars=strike_cols,
        var_name="strike",
        value_name=value_name,
    )
    long_df["strike"] = long_df["strike"].astype(float)
    return long_df


def parse_expiry_from_filename(path):
    """
    Extract expiry as YYYYMMDD from the last 8 digits before '.csv'.
    Example: RTH/2025-04-01_call_bid_20250416.csv -> '20250416'
    """
    m = re.search(r"(\d{8})(?=\.csv$)", os.path.basename(path))
    return m.group(1) if m else None


def load_assets(path):
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    df = df.sort_values("timestamp")
    return df


def load_option_panel(assets_df, call_bid_path, call_ask_path,
                      put_bid_path, put_ask_path):
    """
    Load call/put bid/ask wide-form CSVs and return long-form panel:
    timestamp, expiry, option_type, strike, bid, ask, mid, SPX, T
    """
    # expiry from one of the filenames (they all share the same expiry)
    sample_path = call_bid_path or call_ask_path or put_bid_path or put_ask_path
    expiry_str = parse_expiry_from_filename(sample_path)
    if expiry_str is None:
        raise ValueError(f"Could not parse expiry from {sample_path}")
    expiry = pd.to_datetime(expiry_str, format="%Y%m%d")

    parts = []

    if call_bid_path and call_ask_path:
        cb = pd.read_csv(call_bid_path, parse_dates=[0])
        cb.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
        ca = pd.read_csv(call_ask_path, parse_dates=[0])
        ca.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
        cb_long = melt_option_wide(cb, "bid")
        ca_long = melt_option_wide(ca, "ask")
        calls = cb_long.merge(ca_long, on=["timestamp", "strike"], how="outer")
        calls["option_type"] = "C"
        parts.append(calls)

    if put_bid_path and put_ask_path:
        pb = pd.read_csv(put_bid_path, parse_dates=[0])
        pb.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
        pa = pd.read_csv(put_ask_path, parse_dates=[0])
        pa.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
        pb_long = melt_option_wide(pb, "bid")
        pa_long = melt_option_wide(pa, "ask")
        puts = pb_long.merge(pa_long, on=["timestamp", "strike"], how="outer")
        puts["option_type"] = "P"
        parts.append(puts)

    if not parts:
        return pd.DataFrame()

    opt = pd.concat(parts, ignore_index=True)
    opt["expiry"] = expiry
    opt["mid"] = 0.5 * (opt["bid"] + opt["ask"])

    # merge SPX from assets
    panel = opt.merge(
        assets_df[["timestamp", "SPX"]],
        on="timestamp",
        how="left",
    )
    panel = panel.dropna(subset=["SPX", "mid"])
    # time to maturity in years
    panel["T"] = (panel["expiry"] - panel["timestamp"]).dt.total_seconds() / (365.0 * 24 * 3600)

    # sort for sanity
    panel = panel.sort_values(["timestamp", "option_type", "strike"])
    return panel

def compute_iv_from_args(args):
    """
    Worker for parallel IV computation.
    args = (price, S, K, T, option_type)
    """
    price, S, K, T, opt_type = args
    return implied_vol(
        price=price,
        S=S,
        K=K,
        T=T,
        r=R,
        option_type=opt_type,
        q=Q,
    )

# ----------------------------------------
# Main batch loop
# ----------------------------------------

def process_session_folder(base_dir: str):
    """
    Process all days for a given session folder (RTH or ETH).
    For each *_assets.csv, find corresponding option files and compute IV.
    """
    print(f"=== Processing session folder: {base_dir} ===")

    asset_files = sorted(glob.glob(os.path.join(base_dir, "*_assets.csv")))

    if not asset_files:
        print(f"No asset files found in {base_dir}")
        return

    for asset_path in asset_files:
        trade_date = os.path.basename(asset_path).split("_")[0]  # '2025-04-01'
        print(f"\n--- Trade date {trade_date} ({base_dir}) ---")

        # glob option files for this date
        pattern_base = os.path.join(base_dir, f"{trade_date}_*")
        all_files_for_day = glob.glob(pattern_base + "*.csv")

        # group by expiry (YYYYMMDD at the end)
        by_expiry = {}
        for fpath in all_files_for_day:
            if "_assets" in fpath:
                continue
            exp = parse_expiry_from_filename(fpath)
            if exp is None:
                continue
            by_expiry.setdefault(exp, []).append(fpath)

        if not by_expiry:
            print(f"No option files found for {trade_date} in {base_dir}")
            continue

        assets_df = load_assets(asset_path)

        for expiry_str, files in sorted(by_expiry.items()):
            print(f"  -> Expiry {expiry_str}: {len(files)} files")

            def find(kind):
                # e.g. 2025-04-01_call_bid_20250416.csv
                for f in files:
                    if f"{kind}_" in f:
                        return f
                return None

            call_bid = find("call_bid")
            call_ask = find("call_ask")
            put_bid  = find("put_bid")
            put_ask  = find("put_ask")

            if not any([call_bid, call_ask, put_bid, put_ask]):
                print(f"    No call/put files found for expiry {expiry_str}, skipping.")
                continue

            panel = load_option_panel(assets_df, call_bid, call_ask, put_bid, put_ask)
            if panel.empty:
                print(f"    Panel empty for expiry {expiry_str}, skipping.")
                continue

            # compute IV for each row
            # This may take a bit of time depending on dataset size
            # Prepare arguments as plain tuples (faster + picklable)
            
            print(f"Start computing IVs at {datetime.now().strftime('%H:%M:%S')}")
            start = time.time()

            args_iter = zip(
                panel["mid"].values,
                panel["SPX"].values,
                panel["strike"].values,
                panel["T"].values,
                panel["option_type"].values,
            )

            args_list = list(args_iter)  # materialize once

            # Parallel IV computation
            max_workers = os.cpu_count() or 4

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                ivs = list(executor.map(compute_iv_from_args, args_list, chunksize=1000))

            panel["iv"] = ivs
            
            end = time.time()
            elapsed = end - start
            print(f"Finished at {datetime.now().strftime('%H:%M:%S')}   (took {elapsed:.2f} seconds)")
            
            # TODO: oroginal below. just wanted one liner above
            # ivs = []
            # for _, row in panel.iterrows():
            #     iv = implied_vol(
            #         price=row["mid"],
            #         S=row["SPX"],
            #         K=row["strike"],
            #         T=row["T"],
            #         r=R,
            #         option_type=row["option_type"],
            #         q=Q,
            #     )
            #     ivs.append(iv)

            # panel["iv"] = ivs

            # save out
            out_name = f"IV_{base_dir}_{trade_date}_exp{expiry_str}.csv"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            panel.to_csv(out_path, index=False)
            print(f"Saved IV panel to {out_path}  (rows: {len(panel)})")


def main():
    # for base_dir in BASE_DIRS:
    for base_dir in ["IV_Calc"]:
        if os.path.isdir(base_dir):
            process_session_folder(base_dir)
        else:
            print(f"Warning: directory {base_dir} does not exist, skipping.")


if __name__ == "__main__":
    main()
