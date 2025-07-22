import os
import pandas as pd
import numpy as np
from keras import Sequential
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import csv


# 读取特征数据（新版数据格式）
def load_dataset(file_path):

    df = pd.read_csv(file_path)

    proteins = df.iloc[:, 0]
    features = df.iloc[:, 1:-1]
    labels = df.iloc[:, -1]
    return proteins, features, labels



def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    sn = recall_score(y_true, y_pred)
    sp = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    return acc, mcc, auc, sn, sp, f1



train_file = r"../Dimensionality Reduction/results300/Progen2_train_dim.csv"
test_file = r"../Dimensionality Reduction/results300/Progen2_test_dim.csv"



train_proteins, train_features, train_labels = load_dataset(train_file)
test_proteins, test_features, test_labels = load_dataset(test_file)


numerical_features = train_features.select_dtypes(include=np.number).columns.tolist()
categorical_features = train_features.select_dtypes(include='object').columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)



X_train = preprocessor.fit_transform(train_features)
X_test = preprocessor.transform(test_features)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results = []
best_model = None
best_sn = -1
best_fold = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, train_labels)):


    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]


    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(X_tr.shape[1], 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    history = model.fit(X_tr, y_tr,
                        validation_data=(X_val, y_val),
                        epochs=10,
                        batch_size=32,
                        verbose=1)


    val_pred = model.predict(X_val)
    val_pred_labels = np.argmax(val_pred, axis=1)
    val_true_labels = np.argmax(y_val, axis=1)


    current_sn = recall_score(val_true_labels, val_pred_labels)
    if current_sn > best_sn:
        best_sn = current_sn
        best_model = model
        best_fold = fold + 1


    metrics = compute_metrics(val_true_labels, val_pred_labels, val_pred[:, 1])
    fold_results.append({
        'Fold': fold + 1,
        'Accuracy': metrics[0],
        'MCC': metrics[1],
        'AUC': metrics[2],
        'Sn': metrics[3],
        'Sp': metrics[4],
        'F1': metrics[5]
    })


    print(f"Accuracy: {metrics[0]:.4f} | MCC: {metrics[1]:.4f}")
    print(f"AUC: {metrics[2]:.4f} | F1: {metrics[5]:.4f}")
    print(f"Sensitivity: {metrics[3]:.4f} | Specificity: {metrics[4]:.4f}")


avg_results = {
    'Accuracy': np.mean([f['Accuracy'] for f in fold_results]),
    'MCC': np.mean([f['MCC'] for f in fold_results]),
    'AUC': np.mean([f['AUC'] for f in fold_results]),
    'Sn': np.mean([f['Sn'] for f in fold_results]),
    'Sp': np.mean([f['Sp'] for f in fold_results]),
    'F1': np.mean([f['F1'] for f in fold_results])
}


print(f"Accuracy: {avg_results['Accuracy']:.4f}")
print(f"MCC: {avg_results['MCC']:.4f}")
print(f"AUC: {avg_results['AUC']:.4f}")
print(f"Sensitivity: {avg_results['Sn']:.4f}")
print(f"Specificity: {avg_results['Sp']:.4f}")
print(f"F1 Score: {avg_results['F1']:.4f}")


test_pred = best_model.predict(X_test)
test_pred_labels = np.argmax(test_pred, axis=1)
test_pred_probs = test_pred[:, 1]


results_df = pd.DataFrame({
    'Protein': test_proteins,
    'True_Label': test_labels,
    'Predicted_Label': test_pred_labels,
    'Probability': test_pred_probs
})
results_path = r"predictions.csv"
results_df.to_csv(results_path, index=False)


# 计算测试指标
test_metrics = compute_metrics(test_labels, test_pred_labels, test_pred_probs)


print(f"Accuracy: {test_metrics[0]:.4f}")
print(f"MCC: {test_metrics[1]:.4f}")
print(f"AUC: {test_metrics[2]:.4f}")
print(f"Sensitivity: {test_metrics[3]:.4f}")
print(f"Specificity: {test_metrics[4]:.4f}")
print(f"F1 Score: {test_metrics[5]:.4f}")
