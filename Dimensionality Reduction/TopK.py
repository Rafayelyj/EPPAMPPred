import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os


def load_original_data(pos_path, neg_path):
    pos_df = pd.read_csv(pos_path, index_col=0)
    pos_df['label'] = 1

    neg_df = pd.read_csv(neg_path, index_col=0)
    neg_df['label'] = 0

    return pd.concat([pos_df, neg_df], axis=0)


positive_path = "../Progen2/feature/XU_pretrain_train_positive_features.csv"
negative_path = "../Progen2/feature/XU_pretrain_train_negative_features.csv"
positive_test_path = "../Progen2/feature/XU_AMP_features.csv"
negative_test_path = "../Progen2/feature/XU_nonAMP_features.csv"


original_train_df = load_original_data(positive_path, negative_path)
original_test_df = load_original_data(positive_test_path, negative_test_path)


train_df_shuffled = original_train_df.sample(frac=1, random_state=42)


X_train = train_df_shuffled.drop(columns=['label'])
y_train = train_df_shuffled['label']
X_test = original_test_df.drop(columns=['label'])
y_test = original_test_df['label']

model = XGBClassifier(
    n_estimators=100,
    tree_method='hist',
    random_state=42
)


model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)


feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=False)

TOP_K = 260
selected_features = feature_importance.head(TOP_K)['feature'].tolist()
print(f"Selected top {TOP_K} features:\n{selected_features}")

scaler = StandardScaler()
scaler.fit(X_train[selected_features])


X_train_original = original_train_df[selected_features]
X_train_scaled = scaler.transform(X_train_original)

dim_reduction_train_df = pd.DataFrame(
    X_train_scaled,
    columns=selected_features,
    index=original_train_df.index
).reset_index().rename(columns={'index': 'Protein'})
dim_reduction_train_df['label'] = original_train_df['label'].values


X_test_original = original_test_df[selected_features]
X_test_scaled = scaler.transform(X_test_original)

dim_reduction_test_df = pd.DataFrame(
    X_test_scaled,
    columns=selected_features,
    index=original_test_df.index
).reset_index().rename(columns={'index': 'Protein'})
dim_reduction_test_df['label'] = original_test_df['label'].values


output_train_path = "results260/Progen2_train_dim.csv"
os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
dim_reduction_train_df.to_csv(output_train_path, index=False)



output_test_path = "results260/Progen2_test_dim.csv"
os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
dim_reduction_test_df.to_csv(output_test_path, index=False)


