import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MaxAbsScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Attention, Flatten
from tensorflow.keras.utils import to_categorical
import csv


model_save_path = "../Progen2/models1"
os.makedirs(model_save_path, exist_ok=True)



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



def build_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(units=64, return_sequences=True)(inputs)
    lstm_out = Dropout(0.5)(lstm_out)
    attention_out = Attention()([lstm_out, lstm_out])
    attention_out = Flatten()(attention_out)
    dense_out = Dense(32, activation='relu')(attention_out)
    dropout_out = Dropout(0.5)(dense_out)
    outputs = Dense(2, activation='softmax')(dropout_out)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



positive_file_path = "../Progen2/feature/XU_pretrain_train_positive_features.csv"
negative_file_path = "../Progen2/feature/XU_pretrain_train_negative_features.csv"
positive_test_file_path = "../Progen2/feature/XU_AMP_features.csv"
negative_test_file_path = "../Progen2/feature/XU_nonAMP_features.csv"


positive_features = load_features(positive_file_path)
negative_features = load_features(negative_file_path)
positive_test_features = load_features(positive_test_file_path)
negative_test_features = load_features(negative_test_file_path)


features = pd.concat([positive_features, negative_features], ignore_index=True)
test_features = pd.concat([positive_test_features, negative_test_features], ignore_index=True)


labels = pd.concat([pd.Series([1] * len(positive_features)), pd.Series([0] * len(negative_features))],
                   ignore_index=True)
test_labels = pd.concat([pd.Series([1] * len(positive_test_features)), pd.Series([0] * len(negative_test_features))],
                        ignore_index=True)


sequences = features.iloc[:, 0]
features = features.iloc[:, 1:]


categorical_features = features.select_dtypes(include=['object']).columns
numerical_features = features.select_dtypes(include=['int64', 'float64']).columns


preprocessor = ColumnTransformer(
    transformers=[
        ('num', MaxAbsScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)



features_preprocessed = preprocessor.fit_transform(features)
test_features_preprocessed = preprocessor.transform(test_features)


features_preprocessed = features_preprocessed.reshape(features_preprocessed.shape[0], 1, features_preprocessed.shape[1])
test_features_preprocessed = test_features_preprocessed.reshape(test_features_preprocessed.shape[0], 1,
                                                                test_features_preprocessed.shape[1])


labels_categorical = to_categorical(labels)
test_labels_categorical = to_categorical(test_labels)


n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results = []
best_model = None
best_sn = -1
best_fold = -1


for fold, (train_index, val_index) in enumerate(skf.split(features_preprocessed, labels)):
    print(f"Processing Fold {fold + 1}/{n_splits}")
    X_train, X_val = features_preprocessed[train_index], features_preprocessed[val_index]
    y_train, y_val = labels_categorical[train_index], labels_categorical[val_index]


    model = build_lstm_attention_model(input_shape=(X_train.shape[1], X_train.shape[2]))


    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)


    val_pred = model.predict(X_val)
    val_pred_labels = np.argmax(val_pred, axis=1)
    val_true_labels = np.argmax(y_val, axis=1)


    val_acc, val_mcc, val_auc, val_sn, val_sp, val_f1 = compute_metrics(val_true_labels, val_pred_labels,
                                                                        val_pred[:, 1])
    fold_results.append({
        'Fold': fold + 1,
        'Accuracy': val_acc,
        'MCC': val_mcc,
        'AUC': val_auc,
        'Sn': val_sn,
        'Sp': val_sp,
        'F1': val_f1
    })
    print(f"Fold {fold + 1} Results:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  MCC: {val_mcc:.4f}")
    print(f"  AUC: {val_auc:.4f}")
    print(f"  Sn: {val_sn:.4f}")
    print(f"  Sp: {val_sp:.4f}")
    print(f"  F1: {val_f1:.4f}")


    if val_sn > best_sn:
        best_sn = val_sn
        best_fold = fold + 1
        best_model = model


    model_save_name = os.path.join(model_save_path, f"best_LSTMAttention_fold_{fold + 1}_sn_{val_sn:.4f}.h5")
    model.save(model_save_name)
    print(f"Saved the best model (Fold {fold + 1}, Sn: {val_sn:.4f}) to {model_save_name}")


print("\nAverage Results over 10 Folds:")
avg_results = {
    'Accuracy': np.mean([fold['Accuracy'] for fold in fold_results]),
    'MCC': np.mean([fold['MCC'] for fold in fold_results]),
    'AUC': np.mean([fold['AUC'] for fold in fold_results]),
    'Sn': np.mean([fold['Sn'] for fold in fold_results]),
    'Sp': np.mean([fold['Sp'] for fold in fold_results]),
    'F1': np.mean([fold['F1'] for fold in fold_results])
}
print(f"  Accuracy: {avg_results['Accuracy']:.4f}")
print(f"  MCC: {avg_results['MCC']:.4f}")
print(f"  AUC: {avg_results['AUC']:.4f}")
print(f"  Sn: {avg_results['Sn']:.4f}")
print(f"  Sp: {avg_results['Sp']:.4f}")
print(f"  F1: {avg_results['F1']:.4f}")


if best_model:
    print(f"\nUsing the best model from Fold {best_fold} with Sn: {best_sn:.4f} for independent testing.")
    test_pred = best_model.predict(test_features_preprocessed)
    test_pred_labels = np.argmax(test_pred, axis=1)
    test_true_labels = np.argmax(test_labels_categorical, axis=1)
    test_acc, test_mcc, test_auc, test_sn, test_sp, test_f1 = compute_metrics(test_true_labels, test_pred_labels,
                                                                              test_pred[:, 1])
    print("\nIndependent Test Set Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  MCC: {test_mcc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Sn: {test_sn:.4f}")
    print(f"  Sp: {test_sp:.4f}")
    print(f"  F1: {test_f1:.4f}")


    fieldnames = ['Model', 'Set', 'Accuracy', 'MCC', 'AUC', 'Sn', 'Sp', 'F1']
    file_exists = os.path.isfile('../Progen2/Progen2-results1.csv')
    with open('../Progen2/Progen2-results1.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'Model': 'LSTM-Attention',
            'Set': '10-Fold CV',
            'Accuracy': f"{avg_results['Accuracy']:.4f}",
            'MCC': f"{avg_results['MCC']:.4f}",
            'AUC': f"{avg_results['AUC']:.4f}",
            'Sn': f"{avg_results['Sn']:.4f}",
            'Sp': f"{avg_results['Sp']:.4f}",
            'F1': f"{avg_results['F1']:.4f}"
        })

        writer.writerow({
            'Model': 'LSTM-Attention',
            'Set': 'Independent Test',
            'Accuracy': f"{test_acc:.4f}",
            'MCC': f"{test_mcc:.4f}",
            'AUC': f"{test_auc:.4f}",
            'Sn': f"{test_sn:.4f}",
            'Sp': f"{test_sp:.4f}",
            'F1': f"{test_f1:.4f}"
        })

