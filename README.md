# Russian Word Sense Induction

NLP project for Word Sense Induction on Russian contexts. The task is to
separate examples of the same target word into sense groups using context.

The project is notebook-first. The main notebooks are intentionally written as
linear pipelines without custom helper functions, so each experiment can be read
and run from top to bottom.

## Approach

- Stratified validation split by `word` and `gold_sense_id`.
- Word-wise evaluation with Adjusted Rand Index.
- `random_state=42` for the current notebook runs.
- Models:
  - `word2vec_logreg`: locally trained Word2Vec context embeddings + Logistic Regression per target word.
  - `bert_target_embedding`: RuBERT target-token embeddings + Logistic Regression per target word.

## Current Verified Result

`word2vec.ipynb` with `random_state=42`:

```text
model             macro ARI   weighted ARI
word2vec_logreg     0.0670        -0.0035
```

The BERT notebook is implemented but should be run separately because it needs
to load `cointegrated/rubert-tiny2` and is much slower than the Word2Vec
baseline.

## Project Structure

```text
data/
  train/train.csv
  test/test.csv
BERTWSI.ipynb
word2vec.ipynb
requirements.txt
.gitignore
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Open and run:

- `word2vec.ipynb` for the fast Word2Vec + Logistic Regression baseline.
- `BERTWSI.ipynb` for the RuBERT target-word embedding experiment.
