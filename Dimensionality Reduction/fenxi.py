import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os
import csv
import shap
import lime

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



explainer = shap.Explainer(model.predict_proba, X_train)
shap_values = explainer(X_train)


feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=False)

TOP_K = 300
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

def interpret_analysis(model, X_train, X_test, feature_names, output_dir="../interpretability"):

    os.makedirs(output_dir, exist_ok=True)

    def shap_analysis():

        explainer = shap.Explainer(model.predict_proba, X_train)


        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            explainer(X_train),
            X_train,
            feature_names=feature_names,
            show=False,
            plot_type="bar"
        )
        plt.savefig(f"{output_dir}/shap_global.png", bbox_inches='tight', dpi=300)
        plt.close()


        sample_idx = 42
        plt.figure(figsize=(10, 6))
        shap.force_plot(
            explainer.expected_value,
            explainer(X_test.iloc[[sample_idx]]),
            X_test.iloc[sample_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.savefig(f"{output_dir}/shap_local_sample{sample_idx}.png")
        plt.close()

        pd.DataFrame(
            explainer(X_train),
            columns=feature_names
        ).to_csv(f"{output_dir}/shap_values.csv", index=False)


    def lime_analysis():

        from lime import lime_tabular


        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=['non-AMP', 'AMP'],
            mode='classification',
            discretize_continuous=False
        )


        for i in range(3):
            exp = explainer.explain_instance(
                X_test.iloc[i].values,
                model.predict_proba,
                num_features=20
            )


            exp.save_to_file(f"{output_dir}/lime_explanation_{i}.html")


            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(f"LIME Explanation - Sample {i}")
            plt.savefig(f"{output_dir}/lime_plot_{i}.png", bbox_inches='tight')
            plt.close()

    def partial_dependence_analysis():

        from sklearn.inspection import PartialDependenceDisplay

        top_features = feature_importance.head(3)['feature'].tolist()

        plt.figure(figsize=(15, 5))
