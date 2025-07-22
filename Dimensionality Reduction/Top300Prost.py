import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os


def load_data(file_path, label):
    df = pd.read_csv(file_path, header=0)
    df['label'] = label
    return df


positive_path = "../ProstT5/feature/ProstT5_XU_pretrain_train_positive.csv"
negative_path = "../ProstT5/feature/ProstT5_XU_pretrain_train_negative.csv"
positive_test_path = "../ProstT5/feature/ProstT5_XU_AMP.csv"
negative_test_path = "../ProstT5/feature/ProstT5_XU_nonAMP.csv"


original_train_df = pd.concat([
    load_data(positive_path, label=1),
    load_data(negative_path, label=0)
], ignore_index=True)

train_df_shuffled = original_train_df.sample(frac=1, random_state=42)

original_test_df = pd.concat([
    load_data(positive_test_path, label=1),
    load_data(negative_test_path, label=0)
], ignore_index=True)

X_train = train_df_shuffled.drop(columns=['label']).iloc[:, :1024]  # 取前1024列
y_train = train_df_shuffled['label']
X_test = original_test_df.drop(columns=['label']).iloc[:, :1024]
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
X_original_train = original_train_df.drop(columns=['label']).iloc[:, :1024]
X_original_train_selected = X_original_train[selected_features]
X_original_train_scaled = scaler.fit_transform(X_original_train_selected)


dim_reduction_train_df = pd.DataFrame(
    X_original_train_scaled,
    columns=selected_features
)
dim_reduction_train_df.insert(0, 'Protein', original_train_df.index)
dim_reduction_train_df['label'] = original_train_df['label'].values


output_train_path = "results260/ProstT5_train_dim.csv"
os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
dim_reduction_train_df.to_csv(output_train_path, index=False)


X_test_selected = original_test_df.drop(columns=['label']).iloc[:, :1024][selected_features]
X_test_scaled = scaler.transform(X_test_selected)

dim_reduction_test_df = pd.DataFrame(
    X_test_scaled,
    columns=selected_features
)
dim_reduction_test_df.insert(0, 'Protein', original_test_df.index)
dim_reduction_test_df['label'] = original_test_df['label'].values


output_test_path = "results260/ProstT5_test_dim.csv"
os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
dim_reduction_test_df.to_csv(output_test_path, index=False)
