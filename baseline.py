import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


dev_df = pd.read_csv("data/preprocessed_dataset/dev.csv", header=0, lineterminator='\n')
train_df =pd.read_csv("data/preprocessed_dataset/train.csv", header=0, lineterminator='\n')
test_df = pd.read_csv("data/preprocessed_dataset/test.csv", header=0, lineterminator='\n')

cleaned_train_df = [column.strip() for column in train_df.columns]
train_df.columns = cleaned_train_df

cleaned_dev_df =[column.strip() for column in dev_df.columns]
dev_df.columns = cleaned_dev_df

cleaned_test_df = [column.strip() for column in test_df.columns]
test_df.columns = cleaned_test_df


print(train_df.columns.tolist())
print(dev_df.columns.tolist())
print(test_df.columns.tolist())

X_train_text = train_df['text'].tolist()
y_train_labels = train_df['labels'].tolist()

X_test_text = test_df['text'].tolist()
y_test_labels = test_df['labels'].tolist()

print(X_train_text[5])
print(y_train_labels[5])

print(X_test_text[5])
print(y_test_labels[5])

vectorizer = TfidfVectorizer()
vectorizer.fit(X_train_text)

X_train = vectorizer.transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

lg_classifier = LogisticRegression(random_state = 42)
lg_classifier.fit(X_train, y_train_labels)
lg_pred = lg_classifier.predict(X_test)
accuracy_lg = accuracy_score( y_test_labels , lg_pred)
recall_lg = recall_score(y_test_labels , lg_pred, average='macro')
precision_lg = precision_score(y_test_labels , lg_pred, average='macro')
f1_lg = f1_score(y_test_labels , lg_pred, average='macro')
print(f"accuracy_lg = {accuracy_lg} recall_lg = {recall_lg} precision_lg = {precision_lg} f1_lg = {f1_lg}")

svc_classifier = SVC(kernel = 'linear', random_state = 42)
svc_classifier.fit(X_train, y_train_labels)
svc_pred = svc_classifier.predict(X_test)
accuracy_svc = accuracy_score( y_test_labels , svc_pred)
recall_svc = recall_score(y_test_labels , svc_pred, average='macro')
precision_svc = precision_score(y_test_labels , svc_pred, average='macro')
f1_svc = f1_score(y_test_labels , svc_pred, average='macro')
print(f"accuracy_svc = {accuracy_svc} recall_svc = {recall_svc} precision_svc = {precision_svc} f1_svc = {f1_svc}")

xgb_classifier = XGBClassifier(random_state = 42)
xgb_classifier.fit(X_train, y_train_labels)
xgb_pred = xgb_classifier.predict(X_test)
accuracy_xgb = accuracy_score( y_test_labels , xgb_pred)
recall_xgb = recall_score(y_test_labels , xgb_pred, average='macro')
precision_xgb = precision_score(y_test_labels , xgb_pred, average='macro')
f1_xgb = f1_score(y_test_labels , xgb_pred, average='macro')
print(f"accuracy_xgb = {accuracy_xgb} recall_xgb = {recall_xgb} precision_xgb = {precision_xgb} f1_xgb = {f1_xgb}")