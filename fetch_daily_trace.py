"""
fetch_daily_trace.py -- Pull daily bond-level data from WRDS TRACE Enhanced.

Server-side aggregation via GROUP BY to minimize data transfer.
Output: markov_vol/data/daily_trace.parquet

Uses half-year chunks (Jan-Jun, Jul-Dec) because yearly queries crash WRDS.
Supports resume: if output file exists, skips completed chunks.
"""
import os
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "")


def _read_pgpass_password(username):
    """Read WRDS password from pgpass (Windows %APPDATA% or ~/.pgpass)."""
    candidates = [
        Path(os.environ.get("APPDATA", "")) / "postgresql" / "pgpass.conf",
        Path.home() / ".pgpass",
    ]
    for pgpass in candidates:
        if not pgpass.exists():
            continue
        for line in pgpass.read_text().strip().splitlines():
            parts = line.split(":")
            if len(parts) >= 5 and parts[3] == username:
                return parts[4]
    return None


def _connect():
    """Connect to WRDS non-interactively via pgpass, else fall back to prompt."""
    import wrds
    username = WRDS_USERNAME
    password = _read_pgpass_password(username) if username else None
    if password:
        import sqlalchemy as sa
        uri = (f"postgresql://{username}:{password}"
               f"@wrds-pgdata.wharton.upenn.edu:9737/wrds")
        conn = wrds.Connection(autoconnect=False, wrds_username=username)
        conn.engine = sa.create_engine(
            uri, isolation_level="AUTOCOMMIT",
            connect_args={"sslmode": "require"})
        conn.connection = conn.engine.connect()
        return conn
    return wrds.Connection(wrds_username=username or None)

OUTPUT = Path(__file__).parent / "data" / "daily_trace.parquet"


def make_chunks():
    """
    Generate half-year date chunks (Jan-Jun, Jul-Dec) for 2002-2025.

    Returns
    -------
    chunks : list of (str, str)
        (start_date, end_date) pairs in 'YYYY-MM-DD' format.
    """
    chunks = []
    for year in range(2002, 2026):
        chunks.append((f"{year}-01-01", f"{year}-06-30"))
        chunks.append((f"{year}-07-01", f"{year}-12-31"))
    return chunks


def pull_chunk(db, start, end):
    """
    Pull one half-year chunk of daily bond aggregates from TRACE Enhanced.

    Parameters
    ----------
    db : sqlalchemy engine/connection
    start, end : str
        Date range in 'YYYY-MM-DD' format.

    Returns
    -------
    df : pd.DataFrame
        Columns: cusip, issuer_cusip, date, n_trades, total_volume,
                 vwap, price_min, price_max, avg_yield.
    """
    sql = text(f"""
        SELECT cusip_id AS cusip,
               SUBSTRING(cusip_id, 1, 6) AS issuer_cusip,
               trd_exctn_dt AS date,
               COUNT(*) AS n_trades,
               SUM(entrd_vol_qt) AS total_volume,
               SUM(rptd_pr * entrd_vol_qt) / NULLIF(SUM(entrd_vol_qt), 0) AS vwap,
               MIN(rptd_pr) AS price_min,
               MAX(rptd_pr) AS price_max,
               AVG(yld_pt) AS avg_yield
        FROM wrdsapps_bondret.trace_enhanced_clean
        WHERE trd_exctn_dt >= '{start}' AND trd_exctn_dt <= '{end}'
          AND rptd_pr > 0 AND entrd_vol_qt > 0 AND rpt_side_cd = 'S'
        GROUP BY cusip_id, SUBSTRING(cusip_id, 1, 6), trd_exctn_dt
        ORDER BY trd_exctn_dt, cusip_id
    """)
    with db.engine.connect() as conn:
        result = conn.execute(sql)
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=cols)


def main():
    """
    Pull all chunks with resume support and auto-reconnect.
    """
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    chunks = make_chunks()

    # Resume support: skip completed chunks
    max_existing_date = None
    existing = None
    if OUTPUT.exists():
        print(f"Found existing file: {OUTPUT}")
        existing = pd.read_parquet(OUTPUT)
        existing['date'] = pd.to_datetime(existing['date'])
        max_existing_date = existing['date'].max()
        print(f"  Existing data through {max_existing_date.date()}")
        print(f"  {len(existing):,} rows")

    # Filter chunks to only those after existing data
    if max_existing_date is not None:
        chunks = [(s, e) for s, e in chunks
                  if pd.Timestamp(e) > max_existing_date]
        print(f"  {len(chunks)} chunks remaining")

    if not chunks:
        print("All chunks already downloaded. Nothing to do.")
        return

    db = _connect()
    all_dfs = []

    for i, (start, end) in enumerate(chunks):
        print(f"Chunk {i+1}/{len(chunks)}: {start} to {end} ... ", end="", flush=True)

        retries = 0
        while retries < 3:
            try:
                t0 = time.time()
                df = pull_chunk(db, start, end)
                elapsed = time.time() - t0
                print(f"{len(df):,} rows in {elapsed:.1f}s")
                all_dfs.append(df)
                break
            except Exception as e:
                retries += 1
                print(f"\n  ERROR: {e}")
                if retries < 3:
                    print(f"  Retry {retries}/3 -- sleeping 5s, reconnecting...")
                    time.sleep(5)
                    try:
                        db.close()
                    except Exception:
                        pass
                    try:
                        db = _connect()
                    except Exception as ce:
                        print(f"  Reconnect failed: {ce}")
                        time.sleep(10)
                else:
                    print(f"  FAILED after 3 retries. Skipping chunk.")

    if not all_dfs:
        print("No new data pulled.")
        return

    new_data = pd.concat(all_dfs, ignore_index=True)
    new_data['date'] = pd.to_datetime(new_data['date'])

    # Combine with existing data if resuming
    if existing is not None:
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    # Deduplicate
    combined = combined.drop_duplicates(subset=['cusip', 'date'], keep='last')
    combined = combined.sort_values(['date', 'cusip']).reset_index(drop=True)

    # Save
    combined.to_parquet(OUTPUT, index=False)
    file_size_mb = OUTPUT.stat().st_size / (1024 * 1024)

    # Summary
    print("\n=== Summary ===")
    print(f"Total rows:    {len(combined):,}")
    print(f"Unique bonds:  {combined['cusip'].nunique():,}")
    print(f"Unique issuers:{combined['issuer_cusip'].nunique():,}")
    print(f"Trading days:  {combined['date'].nunique():,}")
    print(f"Date range:    {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"File size:     {file_size_mb:.1f} MB")
    print(f"Saved to:      {OUTPUT}")

    try:
        db.dispose()
    except Exception:
        pass


if __name__ == "__main__":
    main()
