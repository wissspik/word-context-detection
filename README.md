# Russian Word Sense Induction

NLP project for Word Sense Induction on Russian contexts. The task is to
separate examples of the same target word into sense groups using context.

## Approach
- Models:
  - `word2vec_logreg`: locally trained Word2Vec context embeddings + Logistic Regression per target word.
  - `bertwsi_agglomerative`: BERTWSI target-occurrence embeddings + per-word agglomerative clustering with cosine distance.

`BERTWSI.ipynb` contains the BERTWSI experiment: a multilingual BERT encoder
builds contextual target-occurrence vectors, the target wordpiece vectors are
pooled across the last hidden layers, the resulting vectors are normalized, and
senses are induced by clustering occurrences of the same target word. Gold
labels are used only for local validation scoring and for estimating the number
of clusters per word.


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
- `BERTWSI.ipynb` for the BERTWSI clustering experiment.
