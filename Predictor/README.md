# Antimicrobial Peptide Feature Extraction and Prediction System

This is a deep learning-based antimicrobial peptide (AMP) prediction system that integrates multiple protein language models (Progen2, ProstT5, ESM-2) for feature extraction and uses deep learning models for prediction.

## Project Structure

```
Predictor/
├── combined_app.py          # Main program - graphical user interface
├── tiqu.py                  # Feature extraction module
├── add_label_to_combined.py # Label addition tool
├── data/                    # Example data directory
│   ├── XU_AMP.fasta        # AMP sequence examples
│   └── XU_nonAMP.fasta     # Non-AMP sequence examples
├── models/                  # Pre-trained model directory
│   └── best_model_fold_6.h5 # Prediction model
└── feature1/               # Feature file example directory
```

## Features

- **Multi-model feature extraction**: Supports Progen2, ProstT5, and ESM-2 protein language models
- **Intelligent feature selection**: Feature importance analysis based on SHAP values, automatically selecting optimal features
- **Deep learning prediction**: CNN model with attention mechanism for AMP prediction
- **Graphical interface**: User-friendly Tkinter interface, simple and intuitive operation
- **Batch processing**: Supports batch sequence prediction and analysis

## Environment Requirements

### System Requirements
- Python 3.7+
- Windows/Linux/macOS
- CUDA support (optional, for GPU acceleration)

### Python Dependencies
```
pip install tensorflow>=2.8.0
pip install torch>=1.9.0
pip install transformers>=4.20.0
pip install pandas numpy scikit-learn joblib
pip install shap seaborn matplotlib
pip install biopython
pip install fair-esm  # ESM-2 model
```

## Preparing Model Files

#### Progen2 Model
- Download the model from [Progen2 official repository](https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz)
- Place the model files in the `../Progen2/Progen2_small_local/` directory
- Ensure directory structure:
  ```
  ../Progen2/
  └── Progen2_small_local/
      ├── config.json
      ├── pytorch_model.bin
      └── tokenizer.json
  ```

#### ProstT5 Model
- Download the model from [ProstT5 official repository](https://huggingface.co/Rostlab/ProstT5)
- Place the model files in the `../ProstT5/Local/` directory
- Ensure directory structure:
  ```
  ../ProstT5/
  └── Local/
      ├── config.json
      ├── pytorch_model.bin
      └── tokenizer.json
  ```

#### ESM-2 Model
- ESM-2 model will be automatically downloaded from Hugging Face, no manual preparation needed
- Will be cached locally on first use

## Usage Steps

1. **Start the Program**
   ```bash
   python combined_app.py
   ```

2. **Feature Extraction**
   - Click "Browse..." to select FASTA format sequence files
   - Select output directory (default in feature folder)
   - Check required feature extraction models (all recommended)
   - Click "Extract Features" to start extraction
   - Wait for progress to complete (time depends on sequence quantity and model selection)

3. **Prediction Analysis**
   - After feature extraction completes, CSV file path will be automatically filled
   - Confirm model path is correct (models/best_model_fold_6.h5)
   - Click "Start Prediction" to start prediction
   - View prediction results and performance metrics

4. **Save Results**
   - Click "Save Results" to save prediction results
   - Results include: sequence ID, true label, predicted label, positive class probability

## Input File Format

### FASTA Format Requirements
```
>sequence_id_1
MKTLL... (protein sequence)
>sequence_id_2
MKVIL... (protein sequence)
```

### CSV Feature File Format
```csv
ID,feature1,feature2,...,label
seq1,0.123,0.456,...,1
seq2,0.789,0.012,...,0
```

## Output Description

### Feature Extraction Output
- `*_progen2_full.csv`: Complete Progen2 features
- `*_progen2_selected.csv`: Selected Progen2 features (300 dimensions)
- `*_progen2_scaled.csv`: Standardized Progen2 features
- `*_prost5_full.csv`: Complete ProstT5 features
- `*_prost5_selected.csv`: Selected ProstT5 features (300 dimensions)
- `*_prost5_scaled.csv`: Standardized ProstT5 features
- `*_esm.csv`: ESM-2 features
- `*_combined.csv`: Combined all features

### Prediction Output
- **ID**: Sequence identifier
- **True Label**: True label (1=AMP, 0=non-AMP)
- **Predicted Label**: Predicted label (1=AMP, 0=non-AMP)
- **Positive Probability**: Probability of being predicted as AMP

## Notes

1. **Memory Management**
   - Large file processing may require significant memory
   - Recommend batch processing for large sequence quantities (>1000 sequences)

2. **Model Paths**
   - Ensure Progen2 and ProstT5 model paths are correct
   - Missing model files will prevent corresponding feature extraction

3. **GPU Usage**
   - Supports CUDA acceleration, will automatically detect GPU
   - Will automatically use CPU mode without GPU (slower speed)

4. **Cache Mechanism**
   - ESM-2 model will automatically download and cache on first use
   - Subsequent uses will directly load from local cache

5. **Error Handling**
   - Check error messages in the log window
   - Ensure all dependencies are correctly installed
   - Verify input file format is correct 