#!/usr/bin/env python3
"""
Convert UMLS Metathesaurus RRF tables into Neo4j bulk-loader CSVs.

Usage
-----
$ python umls_to_neo4j.py --input /import --out /import/csv --db umls

The defaults match the docker-compose layout:
  /import           → bind-mount of the `umls-data` volume (read-only in neo4j)
  /import/csv       → small writable folder you can mount read-write
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, List

import pandas as pd


# ---------------------------------------------------------------------------
#  utilities
# ---------------------------------------------------------------------------
def read_rrf_iter(
    rrf_path: Path, column_names: List[str],   # without the trailing pipe column
    filters: dict | None = None,
    chunksize: int = 1_000_000,
) -> Iterator[pd.DataFrame]:
    """
    Yield filtered DataFrame chunks so we never load the full 5-GB MRCONSO.

    `filters` is a dict {column: allowed_values}.
    """
    cols = column_names + ["_t"]                          # dummy for trailing '|'
    reader = pd.read_csv(
        rrf_path,
        sep="|",
        names=cols,
        header=None,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
        engine="python",
    )
    for chunk in reader:
        chunk = chunk.drop(columns="_t")
        if filters:
            mask = pd.Series(True, index=chunk.index)
            for col, allowed in filters.items():
                mask &= chunk[col].isin(allowed)
            chunk = chunk[mask]
        if not chunk.empty:
            yield chunk


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  transformation steps
# ---------------------------------------------------------------------------
def build_concepts_csv(input_dir: Path, out_dir: Path) -> Path:
    """MRCONSO + MRSTY → concepts.csv and semantic_types.csv"""
    mrconso_path = input_dir / "MRCONSO.RRF"
    mrsty_path   = input_dir / "MRSTY.RRF"

    # -------- MRCONSO → one English preferred term per CUI
    cols_conso = [
        "CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI",
        "SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"
    ]
    print("• scanning MRCONSO.RRF …")
    frames = []
    for chunk in read_rrf_iter(
        mrconso_path,
        cols_conso,
        filters={"LAT":["ENG"], "TS":["P"], "ISPREF":["Y"]},
        chunksize=500_000,
    ):
        frames.append(chunk[["CUI", "STR"]])

    conso_df = pd.concat(frames, ignore_index=True).drop_duplicates("CUI")

    # -------- MRSTY → semantic types
    cols_sty = ["CUI","TUI","STN","STY","ATUI","CVF"]
    sty_df = pd.read_csv(
        mrsty_path,
        sep="|",
        names=cols_sty + ["_t"],
        header=None,
        dtype=str,
        low_memory=False,
        engine="python",
    ).drop(columns="_t")

    # save both as separate CSVs (helps Neo4j import parallelism)
    ensure_dir(out_dir)
    concepts_csv = out_dir / "concepts.csv"
    semantic_csv = out_dir / "semantic_types.csv"

    conso_df.rename(columns={"STR":"name"}).to_csv(concepts_csv, index=False, escapechar="\\")
    sty_df[["CUI","TUI","STY"]].to_csv(semantic_csv, index=False, escapechar="\\")

    print(f"  → {concepts_csv.name}  ({len(conso_df):,} rows)")
    print(f"  → {semantic_csv.name}  ({len(sty_df):,} rows)")
    return concepts_csv


def build_relationships_csv(input_dir: Path, out_dir: Path) -> Path:
    """MRREL → relationships.csv (concept ↔ concept edges)"""
    mrrel_path = input_dir / "MRREL.RRF"
    cols_rel = [
        "CUI1","AUI1","STYPE1","REL","CUI2","AUI2",
        "STYPE2","RELA","RUI","SRUI","SAB","SL","RG",
        "DIR","SUPPRESS","CVF"
    ]

    print("• scanning MRREL.RRF …")
    chunks = []
    for chunk in read_rrf_iter(
        mrrel_path,
        cols_rel,
        filters={"STYPE1":["CUI"], "STYPE2":["CUI"]},
        chunksize=1_000_000,
    ):
        reltype = chunk["RELA"].fillna(chunk["REL"])
        chunks.append(
            pd.DataFrame(
                {"startId": chunk["CUI1"], "endId": chunk["CUI2"], "type": reltype}
            )
        )

    rel_df = pd.concat(chunks, ignore_index=True).drop_duplicates()
    ensure_dir(out_dir)
    relationships_csv = out_dir / "relationships.csv"
    rel_df.to_csv(relationships_csv, index=False, escapechar="\\")

    print(f"  → {relationships_csv.name}  ({len(rel_df):,} rows)")
    return relationships_csv


# ---------------------------------------------------------------------------
#  main driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transform UMLS RRF tables into Neo4j bulk CSVs."
    )
    ap.add_argument("--input", default="/import", help="RRF source folder")
    ap.add_argument("--out",   default="/import/csv", help="CSV output folder")
    ap.add_argument("--db",    default="umls", help="target Neo4j database name")
    args = ap.parse_args()

    input_dir = Path(args.input)
    out_dir   = Path(args.out)
    db_name   = args.db

    if not input_dir.exists():
        sys.exit(f"[ERROR] {input_dir} does not exist")

    print(f"=== UMLS → Neo4j: processing files in {input_dir} ===")
    build_concepts_csv(input_dir, out_dir)
    build_relationships_csv(input_dir, out_dir)

    # Suggest the import command
    print("\n=== Next step (offline import) ===")
    print(
        f"neo4j-admin database import full {db_name} \\ \n"
        f"  --id-type=STRING \\ \n"
        f"  --nodes=Concept={out_dir/'concepts.csv'} \\ \n"
        f"  --nodes=SemanticType={out_dir/'semantic_types.csv'} \\ \n"
        f"  --relationships=RELATED_TO={out_dir/'relationships.csv'}"
    )


if __name__ == "__main__":
    main()
