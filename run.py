import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Same label order the original authors used (matters — it's what the
# checkpoints were trained to output as class indices 0/1/2)
LABELS = ['severe', 'moderate', 'not depression']

# Load the real test set — note: despite predict.py treating this file as
# label-less, it actually ships with a genuine "Class labels" column
test_df = pd.read_csv('data/original_dataset/test.tsv', sep='\t', header=0)
test_df.columns = ['pid', 'text', 'label_str']
test_df['label_id'] = test_df['label_str'].apply(lambda s: LABELS.index(s))

y_true = test_df['label_id'].values
texts = list(test_df['text'].values)

MODELS_TO_TEST = {
    'RoBERTa-large-depression':   'rafalposwiata/roberta-large-depression',
    'DepRoBERTa-large-depression': 'rafalposwiata/deproberta-large-depression',
}

results = {}
for name, hf_repo in MODELS_TO_TEST.items():
    print(f'\n=== Evaluating {name} ===')
    model = ClassificationModel(
        'roberta',
        hf_repo,
        num_labels=len(LABELS),
        use_cuda=True,   # flip to False if this OOMs on the 1650 — inference
                         # is much lighter than training, so try True first
    )
    predictions, _ = model.predict(texts)

    results[name] = {
        'accuracy':        accuracy_score(y_true, predictions),
        'precision_macro': precision_score(y_true, predictions, average='macro'),
        'recall_macro':    recall_score(y_true, predictions, average='macro'),
        'f1_macro':        f1_score(y_true, predictions, average='macro'),
    }
    print(results[name])

print('\n=== Summary ===')
for name, m in results.items():
    print(f"{name}: F1(macro)={m['f1_macro']:.4f}  P={m['precision_macro']:.4f}  "
          f"R={m['recall_macro']:.4f}  Acc={m['accuracy']:.4f}")