import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_features(file_path):
    return pd.read_csv(file_path)


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    sn = recall_score(y_true, y_pred)
    sp = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    return acc, mcc, auc, sn, sp, f1



def cross_validation(features, labels, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {
        'fold': [],
        'acc': [],
        'mcc': [],
        'auc': [],
        'sn': [],
        'sp': [],
        'f1': [],
        'positive_count': [],
        'negative_count': []
    }

    sequences = features.iloc[:, 0]
    features = features.iloc[:, 1:]


    categorical_features = features.select_dtypes(include=['object']).columns
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    for fold, (train_index, test_index) in enumerate(skf.split(features, labels), 1):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]


        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=42))
        ])

        try:

            pipeline.fit(X_train, y_train)


            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]


            acc, mcc, auc, sn, sp, f1 = compute_metrics(y_test, y_pred, y_prob)


            results['fold'].append(fold)
            results['acc'].append(acc)
            results['mcc'].append(mcc)
            results['auc'].append(auc)
            results['sn'].append(sn)
            results['sp'].append(sp)
            results['f1'].append(f1)
            results['positive_count'].append(y_test.sum())
            results['negative_count'].append((y_test == 0).sum())


            print(f"Fold {fold}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  MCC: {mcc:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  Sn: {sn:.4f}")
            print(f"  Sp: {sp:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  Positive Count: {y_test.sum()}")
            print(f"  Negative Count: {(y_test == 0).sum()}")

        except Exception as e:
            print(f"Error in fold {fold}: {e}")


    results_df = pd.DataFrame(results)
    return results_df


positive_file_path = "feature/XU_pretrain_val_positive_features.csv"
negative_file_path = "feature/XU_pretrain_val_negative_features.csv"


positive_features = load_features(positive_file_path)
negative_features = load_features(negative_file_path)


features = pd.concat([positive_features, negative_features], ignore_index=True)


labels = pd.concat([pd.Series([1] * len(positive_features)), pd.Series([0] * len(negative_features))],
                   ignore_index=True)


results = cross_validation(features, labels)
