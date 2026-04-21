# Russian Word Sense Induction

NLP project for Word Sense Induction on Russian contexts. The task is to
separate examples of the same target word into sense groups using context.

The project follows a notebook-first structure similar to the MTS WSI assignment:
one notebook for the BERT-based solution and one notebook for a fast classical
baseline.

## Approach

- Stratified validation split by `word` and `gold_sense_id`.
- Word-wise evaluation with Adjusted Rand Index.
- Models:
  - `majority`: predicts the most frequent sense for each target word.
  - `tfidf_logreg`: TF-IDF context features + Logistic Regression per target word.
  - `word2vec_logreg`: locally trained Word2Vec context embeddings + Logistic Regression per target word.
  - `bert_target_embedding`: RuBERT target-token embeddings + Logistic Regression.

## Final Result

Fast classical baselines over 5 random validation splits:

```text
model         macro ARI mean +/- std   weighted ARI mean +/- std
tfidf_logreg        0.2353 +/- 0.0206        0.1355 +/- 0.0181
word2vec_logreg     0.1454 +/- 0.0489        0.0594 +/- 0.0374
majority            0.0118 +/- 0.0000        0.0024 +/- 0.0000
```

`tfidf_logreg` is the current best verified result. The local Word2Vec baseline
is weaker but still clearly above the majority baseline. The BERT baseline is
implemented but should be run separately because it requires downloading/loading
the transformer model and is much slower.

## Project Structure

```text
data/
  train/train.csv
  test/test.csv
BERTWSI.ipynb
baseline.ipynb
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

- `baseline.ipynb` for the verified fast classical baselines.
- `BERTWSI.ipynb` for the RuBERT target-word embedding experiment.
