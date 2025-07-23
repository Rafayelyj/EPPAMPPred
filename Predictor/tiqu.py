#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
from Bio import SeqIO
from transformers import AutoModelForCausalLM, T5Tokenizer, T5EncoderModel
from tokenizers import Tokenizer
import esm
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import shap
import joblib
import logging
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def extract_progen2_features(fasta_file, model_path, output_file=None):
    print(f"===== Progen2 Feature Extraction =====")
    print(f"Loading model and tokenizer: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = Tokenizer.from_file(f"{model_path}/tokenizer.json")
    print("Model and tokenizer loaded")
    print(f"Reading FASTA file: {fasta_file}")
    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)
    print("Encoding protein sequences")
    input_ids = [torch.tensor(tokenizer.encode(seq).ids).unsqueeze(0).to(model.device) for seq in sequences]
    print(f"Encoding complete")
    print("Running forward pass to get feature vectors")
    features = []
    model.eval()
    with torch.no_grad():
        for i, ids_tensor in enumerate(input_ids):
            outputs = model(ids_tensor, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            feature_vector = hidden_states[0, 0, :].cpu().numpy()
            features.append(feature_vector)
    features_array = np.array(features)
    if output_file:
        df = pd.DataFrame(features_array, index=ids)
        df.to_csv(output_file, index_label="ID")
        print(f"Progen2 feature vectors saved to {output_file}")
    return features_array, ids

def add_space_to_sequence(sequence):
    return " ".join(sequence)

def parse_fasta(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    protein_name = ""
    sequence = ""
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if protein_name:
                data.append((protein_name, add_space_to_sequence(sequence)))
            protein_name = line[1:]
            sequence = ""
        else:
            sequence += line
    if sequence:
        data.append((protein_name, add_space_to_sequence(sequence)))
    return data

def extract_prost5_features(fasta_file, model_path, output_file=None, batch_size=8):
    print(f"===== ProstT5 Feature Extraction =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model and tokenizer: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).to(device)
    model.eval()
    print("Model and tokenizer loaded")
    sequences_data = parse_fasta(fasta_file)
    sequences = [seq[1] for seq in sequences_data]
    ids = [seq[0] for seq in sequences_data]
    total_sequences = len(sequences)
    inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    dataset = TensorDataset(input_ids, attention_mask)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    all_embeddings = []
    processed_sequences = 0
    for batch_idx, batch in enumerate(data_loader):
        batch_input_ids, batch_attention_mask = batch
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings)
        processed_sequences += len(batch_input_ids)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    features_array = all_embeddings.cpu().numpy()
    if output_file:
        df = pd.DataFrame(features_array, index=ids)
        df.to_csv(output_file, index_label="ID")
        print(f"ProstT5 feature vectors saved to {output_file}")
    return features_array, ids

def extract_esm_features(fasta_file, output_file=None, feature_dim=300, batch_size=8):
    print(f"===== ESM Feature Extraction =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)
    data = [("protein", seq) for seq in sequences]
    print("Loading ESM model...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    all_features = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
            token_representations = results["representations"][6]
        batch_features = token_representations.mean(1).cpu().numpy()
        if batch_features.shape[1] != feature_dim:
            batch_features = np.resize(batch_features, (batch_features.shape[0], feature_dim))
        all_features.append(batch_features)
    features_array = np.vstack(all_features)
    if output_file:
        if 'positive' in output_file:
            label_value = 1
        elif 'negative' in output_file:
            label_value = 0
        else:
            label_value = None
        df = pd.DataFrame(features_array, index=ids)
        if label_value is not None:
            df['label'] = label_value
        df.to_csv(output_file, index_label="ID")
        print(f"ESM feature vectors saved to {output_file}")
    return features_array, ids

def select_features_with_shap(features, feature_names=None, n_top_features=300, output_file=None):
    print(f"===== Selecting Top {n_top_features} Important Features with SHAP =====")
    n_samples = features.shape[0]
    y = np.zeros(n_samples)
    if feature_names is None:
        feature_names = [f'feature_{i + 1}' for i in range(features.shape[1])]
    model = XGBClassifier(
        n_estimators=100,
        tree_method='hist',
        random_state=42
    )
    print("Training XGBClassifier model...")
    model.fit(features, y)
    try:
        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(shap_values).mean(axis=0)
        }).sort_values("importance", ascending=False)
        selected_features = feature_importance.head(n_top_features)['feature'].tolist()
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        print("Using XGBoost built-in feature importance instead...")
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        selected_features = feature_importance.head(n_top_features)['feature'].tolist()
    if output_file:
        feature_importance.to_csv(output_file, index=False)
        print(f"Feature importance saved to {output_file}")
    return selected_features, feature_importance

def extract_features_for_gui(fasta_file, output_dir=None):
    logger.info(f"Start processing file: {fasta_file}")
    if output_dir is None:
        output_dir = "feature"
    create_output_dir(output_dir)
    fasta_basename = os.path.basename(fasta_file).split('.')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    features_csv = os.path.join(output_dir, f"{fasta_basename}_features_{timestamp}.csv")
    try:
        sequences = []
        ids = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
            ids.append(record.id)
        if not sequences:
            logger.error("No valid sequences found")
            return None
        logger.info("Start extracting ESM features")
        data = [("protein", seq) for seq in sequences]
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        batch_size = 8
        all_features = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6])
                token_representations = results["representations"][6]
            batch_features = token_representations.mean(1).cpu().numpy()
            all_features.append(batch_features)
        features_array = np.vstack(all_features)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_array)
        feature_names = [f'feature_{i}' for i in range(scaled_features.shape[1])]
        feature_df = pd.DataFrame(scaled_features, index=ids, columns=feature_names)
        feature_df.to_csv(features_csv, index_label="ID")
        logger.info(f"Features saved as CSV: {features_csv}")
        return features_csv
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    fasta_file = "data/XU_pretrain_val_negative.fasta"
    feature_dir = "feature"
    create_output_dir(feature_dir)
    progen2_model_path = "Progen2_small_local"
    prost5_model_path = "Local"
    fasta_basename = os.path.basename(fasta_file).split('.')[0]
    progen2_output = os.path.join(feature_dir, f"{fasta_basename}_progen2_full.csv")
    prost5_output = os.path.join(feature_dir, f"{fasta_basename}_prost5_full.csv")
    esm_output = os.path.join(feature_dir, f"{fasta_basename}_esm.csv")
    combined_output = os.path.join(feature_dir, f"{fasta_basename}_combined.csv")
    progen2_features, progen2_ids = extract_progen2_features(fasta_file, progen2_model_path, progen2_output)
    progen2_feature_names = [f'progen2_feature_{i + 1}' for i in range(progen2_features.shape[1])]
    progen2_importance_output = os.path.join(feature_dir, f"{fasta_basename}_progen2_importance.csv")
    progen2_df = pd.DataFrame(progen2_features, index=progen2_ids, columns=progen2_feature_names)
    progen2_selected_features, progen2_importance_df = select_features_with_shap(
        progen2_features,
        progen2_feature_names,
        n_top_features=300,
        output_file=progen2_importance_output
    )
    progen2_selected_df = progen2_df[progen2_selected_features]
    scaler = StandardScaler()
    progen2_scaled_features = scaler.fit_transform(progen2_selected_df)
    progen2_selected_output = os.path.join(feature_dir, f"{fasta_basename}_progen2_selected.csv")
    progen2_selected_df.to_csv(progen2_selected_output, index_label="ID")
    print(f"Progen2 selected features saved to {progen2_selected_output}")
    progen2_scaled_output = os.path.join(feature_dir, f"{fasta_basename}_progen2_scaled.csv")
    pd.DataFrame(progen2_scaled_features, index=progen2_ids, columns=progen2_selected_features).to_csv(
        progen2_scaled_output, index_label="ID"
    )
    print(f"Progen2 scaled features saved to {progen2_scaled_output}")
    joblib.dump(scaler, os.path.join(feature_dir, f"{fasta_basename}_progen2_scaler.pkl"))
    prost5_features, prost5_ids = extract_prost5_features(fasta_file, prost5_model_path, prost5_output)
    prost5_feature_names = [f'prost5_feature_{i + 1}' for i in range(prost5_features.shape[1])]
    prost5_importance_output = os.path.join(feature_dir, f"{fasta_basename}_prost5_importance.csv")
    prost5_df = pd.DataFrame(prost5_features, index=prost5_ids, columns=prost5_feature_names)
    prost5_selected_features, prost5_importance_df = select_features_with_shap(
        prost5_features,
        prost5_feature_names,
        n_top_features=300,
        output_file=prost5_importance_output
    )
    prost5_selected_df = prost5_df[prost5_selected_features]
    scaler = StandardScaler()
    prost5_scaled_features = scaler.fit_transform(prost5_selected_df)
    prost5_selected_output = os.path.join(feature_dir, f"{fasta_basename}_prost5_selected.csv")
    prost5_selected_df.to_csv(prost5_selected_output, index_label="ID")
    print(f"ProstT5 selected features saved to {prost5_selected_output}")
    prost5_scaled_output = os.path.join(feature_dir, f"{fasta_basename}_prost5_scaled.csv")
    pd.DataFrame(prost5_scaled_features, index=prost5_ids, columns=prost5_selected_features).to_csv(
        prost5_scaled_output, index_label="ID"
    )
    print(f"ProstT5 scaled features saved to {prost5_scaled_output}")
    joblib.dump(scaler, os.path.join(feature_dir, f"{fasta_basename}_prost5_scaler.pkl"))
    esm_features, esm_ids = extract_esm_features(fasta_file, esm_output)
    if len(set([tuple(progen2_ids), tuple(prost5_ids), tuple(esm_ids)])) > 1:
        print("Warning: Feature IDs extracted by different models are inconsistent, using the IDs from the first model")
    print("===== Merging Selected Features =====")
    combined_df = pd.DataFrame(index=progen2_ids)
    for i, feature in enumerate(progen2_selected_features):
        combined_df[feature] = progen2_scaled_features[:, i]
    for i, feature in enumerate(prost5_selected_features):
        combined_df[feature] = prost5_scaled_features[:, i]
    esm_feature_names = [f'esm_feature_{i + 1}' for i in range(esm_features.shape[1])]
    esm_df = pd.DataFrame(esm_features, index=esm_ids, columns=esm_feature_names)
    scaler = StandardScaler()
    esm_scaled_features = scaler.fit_transform(esm_df)
    for i, feature in enumerate(esm_feature_names):
        combined_df[feature] = esm_scaled_features[:, i]
    combined_df.to_csv(combined_output, index_label="ID")
    print(f"Merged features saved to: {combined_output}")
    print(f"Merged feature dimension: {combined_df.shape}")
    print("Feature extraction, selection, and merging completed!")

if __name__ == "__main__":
    main()