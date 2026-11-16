"""Count report characters without spaces, as the 02807 limit does (12,500 for a group of 5).

Usage:  python scripts/count_chars.py report/report.txt [--budget 12500]
Works on a .txt / .md / .tex export. Sections are split on lines starting with "#" (Markdown)
or "\\section" (LaTeX) and counted separately, so each writer can check their own budget.
"""
import argparse
import re
from pathlib import Path


def count(text: str) -> int:
    return len(re.sub(r"\s", "", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--budget", type=int, default=12_500)
    args = ap.parse_args()
    text = args.file.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(?=#|\\section)", text)
    for part in parts:
        title = part.strip().splitlines()[0][:60] if part.strip() else "(start)"
        print(f"{count(part):>7,}  {title}")
    total = count(text)
    print(f"{total:>7,}  TOTAL / budget {args.budget:,} -> {'OK' if total <= args.budget else 'TOO LONG by ' + format(total - args.budget, ',')}")


if __name__ == "__main__":
    main()
