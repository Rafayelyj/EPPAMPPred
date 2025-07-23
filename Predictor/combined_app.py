#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
import threading
import traceback
import importlib.util
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def flush(self):
        pass

def get_custom_objects():
    return {
        'MultiHeadAttention': MultiHeadAttention,
        'LayerNormalization': LayerNormalization
    }

def load_features(file_path):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return None
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        return None

class CombinedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Antimicrobial Peptide Feature Extraction and Prediction System")
        self.root.geometry("900x800")
        try:
            spec = importlib.util.spec_from_file_location("tiqu", "tiqu.py")
            self.tiqu_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.tiqu_module)
            logger.info("Successfully imported tiqu.py module")
        except Exception as e:
            logger.error(f"Failed to import tiqu.py module: {e}")
            messagebox.showerror("Error", f"Failed to import tiqu.py module: {e}")
            self.root.destroy()
            return
        self.fasta_file_path = None
        self.data_file_path = None
        self.model_path = "models/best_model_fold_6.h5"
        self.results_df = None
        self.extracted_feature_path = None
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_main_page()
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
    def create_main_page(self):
        main_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(main_frame, text="Feature Extraction and Prediction")
        extract_frame = ttk.LabelFrame(main_frame, text="Feature Extraction", padding=10)
        extract_frame.pack(fill=tk.X, pady=5)
        fasta_frame = ttk.Frame(extract_frame)
        fasta_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fasta_frame, text="FASTA File:").pack(side=tk.LEFT, padx=5)
        self.fasta_entry = ttk.Entry(fasta_frame, width=60)
        self.fasta_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(fasta_frame, text="Browse...", command=self.browse_fasta).pack(side=tk.LEFT, padx=5)
        options_frame = ttk.Frame(extract_frame)
        options_frame.pack(fill=tk.X, pady=5)
        self.use_progen2_var = tk.BooleanVar(value=True)
        self.progen2_path_var = tk.StringVar(value="../Progen2/Progen2_small_local")
        self.use_prost5_var = tk.BooleanVar(value=True)
        self.prost5_path_var = tk.StringVar(value="../ProstT5/Local")
        self.use_esm_var = tk.BooleanVar(value=True)
        output_frame = ttk.Frame(extract_frame)
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        self.output_dir_var = tk.StringVar(value="feature")
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=60)
        output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_dir).pack(side=tk.LEFT, padx=5)
        ttk.Button(extract_frame, text="Extract Features", command=self.start_extraction, style="Accent.TButton").pack(anchor=tk.E, pady=5)
        predict_frame = ttk.LabelFrame(main_frame, text="Prediction", padding=10)
        predict_frame.pack(fill=tk.X, pady=10)
        csv_frame = ttk.Frame(predict_frame)
        csv_frame.pack(fill=tk.X, pady=5)
        ttk.Label(csv_frame, text="Feature CSV File:").pack(side=tk.LEFT, padx=5)
        self.csv_entry = ttk.Entry(csv_frame, width=60)
        self.csv_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(csv_frame, text="Browse...", command=self.browse_csv).pack(side=tk.LEFT, padx=5)
        model_frame = ttk.Frame(predict_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Model File:").pack(side=tk.LEFT, padx=5)
        self.model_entry = ttk.Entry(model_frame, width=60)
        self.model_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.model_entry.insert(0, self.model_path)
        ttk.Button(model_frame, text="Browse...", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        button_frame = ttk.Frame(predict_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Start Prediction", command=self.start_prediction).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.tree = ttk.Treeview(result_frame, columns=("ID", "True Label", "Predicted Label", "Positive Probability"), show="headings")
        self.tree.heading("ID", text="ID")
        self.tree.heading("True Label", text="True Label")
        self.tree.heading("Predicted Label", text="Predicted Label")
        self.tree.heading("Positive Probability", text="Positive Probability")
        self.tree.column("ID", width=150)
        self.tree.column("True Label", width=100)
        self.tree.column("Predicted Label", width=100)
        self.tree.column("Positive Probability", width=100)
        tree_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=8)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scroll.set, state=tk.DISABLED)
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    
    def browse_fasta(self):
        file_path = filedialog.askopenfilename(
            title="Select FASTA File",
            filetypes=[("FASTA Files", "*.fasta *.fa"), ("All Files", "*.*")]
        )
        if file_path:
            self.fasta_file_path = file_path
            self.fasta_entry.delete(0, tk.END)
            self.fasta_entry.insert(0, file_path)
    
    def browse_output_dir(self):
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if output_dir:
            self.output_dir_var.set(output_dir)
    
    def browse_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls"), ("All Files", "*.*")]
        )
        if file_path:
            self.data_file_path = file_path
            self.csv_entry.delete(0, tk.END)
            self.csv_entry.insert(0, file_path)
    
    def browse_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")]
        )
        if model_path:
            self.model_path = model_path
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, model_path)
    
    def start_extraction(self):
        if not self.fasta_file_path:
            messagebox.showerror("Error", "Please select a FASTA file first")
            return
        if not os.path.exists(self.fasta_file_path):
            messagebox.showerror("Error", f"File does not exist: {self.fasta_file_path}")
            return
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.old_stdout = sys.stdout
        sys.stdout = RedirectText(self.log_text)
        self.status_var.set("Extracting features...")
        threading.Thread(target=self.run_extraction, daemon=True).start()
    
    def run_extraction(self):
        try:
            print(f"=== Feature Extraction Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"FASTA File: {self.fasta_file_path}")
            output_dir = self.output_dir_var.get()
            self.tiqu_module.create_output_dir(output_dir)
            print(f"Output Directory: {output_dir}")
            progen2_model_path = self.progen2_path_var.get() if self.use_progen2_var.get() else None
            prost5_model_path = self.prost5_path_var.get() if self.use_prost5_var.get() else None
            use_esm = self.use_esm_var.get()
            fasta_basename = os.path.basename(self.fasta_file_path).split('.')[0]
            progen2_features = None
            progen2_ids = None
            prost5_features = None
            prost5_ids = None
            esm_features = None
            esm_ids = None
            if progen2_model_path:
                progen2_output = os.path.join(output_dir, f"{fasta_basename}_progen2_full.csv")
                print(f"\nExtracting Progen2 features (model: {progen2_model_path})...")
                progen2_features, progen2_ids = self.tiqu_module.extract_progen2_features(
                    self.fasta_file_path, progen2_model_path, progen2_output
                )
                progen2_feature_names = [f'progen2_feature_{i + 1}' for i in range(progen2_features.shape[1])]
                progen2_importance_output = os.path.join(output_dir, f"{fasta_basename}_progen2_importance.csv")
                progen2_df = self.tiqu_module.pd.DataFrame(progen2_features, index=progen2_ids,
                                                       columns=progen2_feature_names)
                progen2_selected_features, progen2_importance_df = self.tiqu_module.select_features_with_shap(
                    progen2_features, progen2_feature_names, n_top_features=300, output_file=progen2_importance_output
                )
                progen2_selected_df = progen2_df[progen2_selected_features]
                scaler = self.tiqu_module.StandardScaler()
                progen2_scaled_features = scaler.fit_transform(progen2_selected_df)
                progen2_selected_output = os.path.join(output_dir, f"{fasta_basename}_progen2_selected.csv")
                progen2_selected_df.to_csv(progen2_selected_output, index_label="ID")
                print(f"Selected Progen2 features saved to {progen2_selected_output}")
                progen2_scaled_output = os.path.join(output_dir, f"{fasta_basename}_progen2_scaled.csv")
                self.tiqu_module.pd.DataFrame(progen2_scaled_features, index=progen2_ids,
                                          columns=progen2_selected_features).to_csv(
                    progen2_scaled_output, index_label="ID"
                )
                print(f"Standardized Progen2 features saved to {progen2_scaled_output}")
                self.tiqu_module.joblib.dump(scaler, os.path.join(output_dir, f"{fasta_basename}_progen2_scaler.pkl"))
            if prost5_model_path:
                prost5_output = os.path.join(output_dir, f"{fasta_basename}_prost5_full.csv")
                print(f"\nExtracting ProstT5 features (model: {prost5_model_path})...")
                prost5_features, prost5_ids = self.tiqu_module.extract_prost5_features(
                    self.fasta_file_path, prost5_model_path, prost5_output
                )
                prost5_feature_names = [f'prost5_feature_{i + 1}' for i in range(prost5_features.shape[1])]
                prost5_importance_output = os.path.join(output_dir, f"{fasta_basename}_prost5_importance.csv")
                prost5_df = self.tiqu_module.pd.DataFrame(prost5_features, index=prost5_ids, columns=prost5_feature_names)
                prost5_selected_features, prost5_importance_df = self.tiqu_module.select_features_with_shap(
                    prost5_features, prost5_feature_names, n_top_features=300, output_file=prost5_importance_output
                )
                prost5_selected_df = prost5_df[prost5_selected_features]
                scaler = self.tiqu_module.StandardScaler()
                prost5_scaled_features = scaler.fit_transform(prost5_selected_df)
                prost5_selected_output = os.path.join(output_dir, f"{fasta_basename}_prost5_selected.csv")
                prost5_selected_df.to_csv(prost5_selected_output, index_label="ID")
                print(f"Selected ProstT5 features saved to {prost5_selected_output}")
                prost5_scaled_output = os.path.join(output_dir, f"{fasta_basename}_prost5_scaled.csv")
                self.tiqu_module.pd.DataFrame(prost5_scaled_features, index=prost5_ids,
                                          columns=prost5_selected_features).to_csv(
                    prost5_scaled_output, index_label="ID"
                )
                print(f"Standardized ProstT5 features saved to {prost5_scaled_output}")
                self.tiqu_module.joblib.dump(scaler, os.path.join(output_dir, f"{fasta_basename}_prost5_scaler.pkl"))
            if use_esm:
                esm_output = os.path.join(output_dir, f"{fasta_basename}_esm.csv")
                print(f"\nExtracting ESM features...")
                esm_features, esm_ids = self.tiqu_module.extract_esm_features(
                    self.fasta_file_path, esm_output
                )
                if esm_features is not None:
                    esm_feature_names = [f'esm_feature_{i + 1}' for i in range(esm_features.shape[1])]
                    esm_df = self.tiqu_module.pd.DataFrame(esm_features, index=esm_ids, columns=esm_feature_names)
                    scaler = self.tiqu_module.StandardScaler()
                    esm_scaled_features = scaler.fit_transform(esm_df)
                    esm_scaled_output = os.path.join(output_dir, f"{fasta_basename}_esm_scaled.csv")
                    self.tiqu_module.pd.DataFrame(esm_scaled_features, index=esm_ids,
                                              columns=esm_feature_names).to_csv(
                        esm_scaled_output, index_label="ID"
                    )
                    print(f"Standardized ESM features saved to {esm_scaled_output}")
                    self.tiqu_module.joblib.dump(scaler, os.path.join(output_dir, f"{fasta_basename}_esm_scaler.pkl"))
            if (progen2_features is not None or prost5_features is not None or esm_features is not None):
                print("\n===== Merging Selected Features =====")
                reference_ids = None
                if progen2_ids is not None:
                    reference_ids = progen2_ids
                elif prost5_ids is not None:
                    reference_ids = prost5_ids
                elif esm_ids is not None:
                    reference_ids = esm_ids
                if reference_ids is not None:
                    combined_df = self.tiqu_module.pd.DataFrame(index=reference_ids)
                    if progen2_features is not None:
                        for i, feature in enumerate(progen2_selected_features):
                            combined_df[feature] = progen2_scaled_features[:, i]
                    if prost5_features is not None:
                        for i, feature in enumerate(prost5_selected_features):
                            combined_df[feature] = prost5_scaled_features[:, i]
                    if esm_features is not None:
                        for i, feature in enumerate(esm_feature_names):
                            combined_df[feature] = esm_scaled_features[:, i]
                    combined_output = os.path.join(output_dir, f"{fasta_basename}_combined.csv")
                    combined_df.to_csv(combined_output, index_label="ID")
                    print(f"Merged features saved to: {combined_output}")
                    print(f"Merged feature shape: {combined_df.shape}")
                    try:
                        import subprocess
                        import sys
                        subprocess.run([sys.executable, "add_label_to_combined.py", combined_output], check=True)
                        print(f"Label column added to {combined_output}")
                    except Exception as e:
                        print(f"Failed to call add_label_to_combined.py to add label column: {e}")
                    self.extracted_feature_path = combined_output
                    self.root.after(0, lambda: self.csv_entry.delete(0, tk.END))
                    self.root.after(0, lambda: self.csv_entry.insert(0, combined_output))
                    self.data_file_path = combined_output
                model_dir = os.path.join(output_dir, "models")
                self.tiqu_module.create_output_dir(model_dir)
                if progen2_features is not None:
                    self.tiqu_module.joblib.dump({
                        'selected_features': progen2_selected_features,
                        'importance_df': progen2_importance_df
                    }, os.path.join(model_dir, "progen2_feature_selector.pkl"))
                if prost5_features is not None:
                    self.tiqu_module.joblib.dump({
                        'selected_features': prost5_selected_features,
                        'importance_df': prost5_importance_df
                    }, os.path.join(model_dir, "prost5_feature_selector.pkl"))
            print(f"\n=== Feature Extraction Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))
            sys.stdout = self.old_stdout
    
    def start_prediction(self):
        if not self.data_file_path:
            messagebox.showerror("Error", "Please select a CSV file first")
            return
        if not os.path.exists(self.data_file_path):
            messagebox.showerror("Error", f"File does not exist: {self.data_file_path}")
            return
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model file does not exist: {self.model_path}")
            return
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.old_stdout = sys.stdout
        sys.stdout = RedirectText(self.log_text)
        self.status_var.set("Predicting...")
        threading.Thread(target=self.run_prediction, daemon=True).start()
    
    def run_prediction(self):
        try:
            tf.get_logger().setLevel('ERROR')
            print(f"=== Prediction Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"Feature File: {self.data_file_path}")
            print(f"Model File: {self.model_path}")
            data = load_features(self.data_file_path)
            if data is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to read data file"))
                self.root.after(0, lambda: self.status_var.set("Ready"))
                return
            ids = data.iloc[:, 0].values
            if data.shape[1] > 2:
                labels = data.iloc[:, -1].values
                features = data.iloc[:, 1:-1]
            else:
                labels = np.zeros(data.shape[0])
                features = data.iloc[:, 1:]
            features_array = features.values.astype(float)
            if np.isnan(features_array).any():
                features_array = np.nan_to_num(features_array, nan=0.0)
            if np.isinf(features_array).any():
                features_array = np.nan_to_num(features_array, posinf=1e10, neginf=-1e10)
            features_preprocessed = features_array.reshape(features_array.shape[0], features_array.shape[1], 1)
            model = None
            try:
                custom_objects = get_custom_objects()
                with tf.keras.utils.custom_object_scope(custom_objects):
                    model = load_model(self.model_path)
            except Exception as e:
                print(f" ")
            if model is None:
                try:
                    input_dim = features_preprocessed.shape[1]
                    input_layer = tf.keras.layers.Input(shape=(input_dim, 1))
                    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
                    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
                    attention_output = MultiHeadAttention(
                        num_heads=4,
                        key_dim=64
                    )(
                        query=pool1,
                        key=pool1,
                        value=pool1
                    )
                    attention_output = tf.keras.layers.Dropout(0.2)(attention_output)
                    attention_output = LayerNormalization()(attention_output + pool1)
                    flat = tf.keras.layers.Flatten()(attention_output)
                    dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
                    output_layer = tf.keras.layers.Dense(2, activation='softmax')(dense1)
                    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    model.load_weights(self.model_path)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {str(e)}"))
                    self.root.after(0, lambda: self.status_var.set("Ready"))
                    return
            predictions = model.predict(features_preprocessed, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            prediction_probs = predictions[:, 1]
            self.results_df = pd.DataFrame({
                'ID': ids,
                'True Label': labels,
                'Predicted Label': predicted_classes,
                'Positive Probability': np.round(prediction_probs, 4)
            })
            accuracy = np.mean(predicted_classes == labels)
            true_positives = np.sum((predicted_classes == 1) & (labels == 1))
            true_negatives = np.sum((predicted_classes == 0) & (labels == 0))
            false_positives = np.sum((predicted_classes == 1) & (labels == 0))
            false_negatives = np.sum((predicted_classes == 0) & (labels == 1))
            sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            print(f"\n=== Prediction Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"Accuracy: {accuracy:.4f}")
            self.root.after(0, lambda: self.update_results(accuracy, sensitivity, specificity,
                                                         true_positives, true_negatives,
                                                         false_positives, false_negatives))
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.root.after(0, lambda: self.status_var.set("Ready"))
            sys.stdout = self.old_stdout
    
    def update_results(self, accuracy, sensitivity, specificity, true_positives, true_negatives, false_positives, false_negatives):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, row in self.results_df.iterrows():
            self.tree.insert("", tk.END, values=(row['ID'], row['True Label'], row['Predicted Label'], row['Positive Probability']))
        sys.stdout = self.old_stdout
    
    def save_results(self):
        if self.results_df is None:
            messagebox.showerror("Error", "No prediction results to save")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Prediction Results",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.results_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Prediction results saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving results: {str(e)}")

def main():
    root = tk.Tk()
    app = CombinedApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 