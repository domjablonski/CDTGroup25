#!/usr/bin/env python3
"""Near-duplicate removal for parquet files using MinHash + LSH.

Usage:
    python minhash.py                                  # looks in ../data next to this script
    python minhash.py --data-dir /path/to/data
    python minhash.py --columns name description steps
    python minhash.py --threshold 0.7 --num-perm 256

pip install datasketch pyarrow
"""
import argparse
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

HERE = Path(__file__).resolve().parent
CANDIDATE_COLUMNS = ["name", "description", "ingredients", "steps", "text"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "data",
                    help="folder searched recursively for parquet files (default: ../data)")
    ap.add_argument("--pattern", default="recipes.parquet",
                    help="glob pattern for input files (default: recipes.parquet)")
    ap.add_argument("--output", type=Path, default=HERE / "deduped.parquet")
    ap.add_argument("--columns", nargs="+", default=None,
                    help="columns to hash (default: auto-detect name/description/ingredients/steps/text)")
    ap.add_argument("--threshold", type=float, default=0.8, help="Jaccard threshold for duplicates")
    ap.add_argument("--num-perm", type=int, default=128, help="MinHash permutations")
    ap.add_argument("--ngram", type=int, default=5, help="word n-gram size for shingles")
    ap.add_argument("--batch-size", type=int, default=10_000)
    return ap.parse_args()


def resolve_data_dir(path: Path) -> Path:
    """Fall back to any sibling folder starting with 'data' (handles e.g. 'data ' with a trailing space)."""
    if path.is_dir():
        return path
    for cand in sorted(path.parent.glob("data*")):
        if cand.is_dir():
            print(f"'{path}' not found, using '{cand}' instead")
            return cand
    return path


def to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(to_text(v) for v in value)
    if hasattr(value, "tolist"):
        return to_text(value.tolist())
    return str(value)


def shingles(text: str, n: int):
    words = re.findall(r"\w+", text.lower())
    if len(words) <= n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def make_minhash(text: str, num_perm: int, n: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles(text, n):
        m.update(s.encode("utf8"))
    return m


def pick_columns(schema: pa.Schema, requested):
    names = schema.names
    if requested:
        missing = [c for c in requested if c not in names]
        if missing:
            raise SystemExit(f"Columns not found: {missing}\nAvailable: {names}")
        return requested
    cols = [c for c in CANDIDATE_COLUMNS if c in names]
    if cols:
        return cols
    # fallback: every string or list-of-string column
    def is_texty(t):
        return pa.types.is_string(t) or pa.types.is_large_string(t) or (
            (pa.types.is_list(t) or pa.types.is_large_list(t))
            and (pa.types.is_string(t.value_type) or pa.types.is_large_string(t.value_type)))
    cols = [f.name for f in schema if is_texty(f.type)]
    if not cols:
        raise SystemExit(f"No text columns found. Available: {names}")
    return cols


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    files = sorted(data_dir.rglob(args.pattern))
    if not files:
        sys.exit(f"No files matching '{args.pattern}' under {data_dir}")
    print("Input files:", *files, sep="\n  ")

    schema = pq.ParquetFile(files[0]).schema_arrow
    cols = pick_columns(schema, args.columns)
    print("Hashing columns:", cols)
    print("All columns:", schema.names)

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    writer = None
    doc_id = kept = dropped = 0

    try:
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=args.batch_size):
                table = pa.Table.from_batches([batch])
                col_data = [table.column(c).to_pylist() for c in cols]
                keep_mask = []
                for row in zip(*col_data):
                    text = " ".join(to_text(v) for v in row)
                    m = make_minhash(text, args.num_perm, args.ngram)
                    if lsh.query(m):
                        keep_mask.append(False)
                        dropped += 1
                    else:
                        lsh.insert(str(doc_id), m)
                        keep_mask.append(True)
                        kept += 1
                    doc_id += 1
                out = table.filter(pa.array(keep_mask))
                if writer is None:
                    writer = pq.ParquetWriter(args.output, out.schema)
                writer.write_table(out)
                print(f"processed {doc_id} docs: kept {kept}, dropped {dropped}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        sys.exit("Nothing was written: input files contained no rows")
    print(f"Done. kept {kept}, dropped {dropped} -> {args.output}")


if __name__ == "__main__":
    main()