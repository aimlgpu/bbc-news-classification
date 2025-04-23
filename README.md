# BBC News Classification Kaggle Mini-Project
Classifies BBC News articles into five categories using supervised (Logistic Regression, Naive Bayes) and unsupervised (NMF) learning.

## Files
- `bbc_news_classification_final.ipynb`: Main notebook with EDA, modeling, and comparison.
- `data/`: Datasets (`BBC News Train.csv`, `BBC News Test.csv`, `BBC News Sample Solution.csv`).
- `nltk_data/`: NLTK resources (`punkt`, `stopwords`, `wordnet`).
- Plots: `category_distribution.png`, `text_length_distribution.png`, `common_words.png`, `nmf_confusion_matrix.png`.
- Models: `tfidf_vectorizer.pkl`, `lr_model.pkl`, `nb_model.pkl`.
- Outputs: `topic_assignments.csv`, `submission.csv`, `submission_evaluation.csv`.

## Setup
```bash
pip install pandas matplotlib nltk scikit-learn
