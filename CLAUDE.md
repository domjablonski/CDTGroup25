# Food.com project (Python 3.11, pandas, NumPy, scikit-learn, networkx) - DTU 02807
Topic: "Ingredient pairing and recipe similarity on Food.com".
Contract (frozen files, schemas, behaviour, ownership, tests): @docs/CONTRACT.md
Stories per role and sprint: docs/PROJECT_INSTRUCTIONS_FOR_CLAUDE_CODE.txt
Team: R1 data + ingredients + reproducibility - R2 MinHash/LSH (RQ2) - R3 PCY/A-Priori (RQ1) -
R4 PMI ingredient graph + Louvain (RQ3) - R5 calorie classifier from LSH neighbours (RQ4) + report editor.

## Commands (repo root, venv active)
- Install: `pip install -r requirements.txt && pip install -e .`
- Data: `python -m foodcom.download` then `python -m foodcom.cleaning`
- All tests: `pytest -q` - fast only: `pytest -m "not slow"` - one file: `pytest tests/test_lsh.py -q`
- Full pipeline: `python scripts/run_all.py` - notebooks: `jupyter lab notebooks/`

## Rules
- IMPORTANT: never edit src/foodcom/config.py, src/foodcom/schemas.py, tests/conftest.py,
  tests/fixtures/, tests/test_contract.py, docs/CONTRACT.md. If the contract must change, STOP
  and explain why. Never change a frozen function signature or docstring.
- Only edit the files owned by the role of the current story (CONTRACT 3.1). Work on the story
  branch, never on main.
- Tests first (toy examples with a known answer), see them fail, then implement.
- NEVER weaken, skip or delete a test or an assert to make it pass.
- MinHash/LSH and A-Priori/PCY are our own code; datasketch/mlxtend only in comparison tests.
- No hard-coded paths, seeds or column lists: use config and schemas; schemas.validate() before
  saving an exchanged file. Develop on load_*(sample=True); full-data tests are @pytest.mark.data.
- RQ4: only training neighbours vote; copies (DUPLICATES) never split between train and test.
- Every number for the report is written to results/*.csv by code in the repo.
- After each change run pytest and show the summary line before claiming success.
- 3 failed fixes on the same problem: stop, list 3 causes, test each with a tiny experiment.
- Commit on the story branch when green. Never push without asking.
  Append to docs/AI_USAGE.md at the end of each story.
