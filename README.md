# Russian Word Sense Induction

NLP project for Word Sense Induction on Russian contexts. The task is to
separate examples of the same target word into sense groups using context.

## Approach
- Models:
  - `word2vec_logreg`: locally trained Word2Vec context embeddings + Logistic Regression per target word.
  - `bert_target_embedding`: RuBERT target-token embeddings + Logistic Regression per target word.


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
