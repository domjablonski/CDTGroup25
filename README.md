# CDTGroup25
test
Another test

## Data science environment

Create and activate a project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv/bin/python` as the Python interpreter. To run the
preprocessing script, activate the environment and run:

```bash
python preprocessing/preprocessing.py
```

The script locates `RAW_recipes.csv` and `RAW_interactions.csv` in a `data*`
directory under the project root, so it can also use the provided nested
dataset folder.