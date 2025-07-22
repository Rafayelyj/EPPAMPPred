import os
import pandas as pd
import numpy as np
from Bio import SeqIO

def read_file(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def aac_feature(sequences):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    features = []

    for seq in sequences:
        seq = seq.upper()
        seq_length = len(seq)
        aac_vector = [seq.count(aa) / seq_length for aa in amino_acids]
        features.append(aac_vector)

    return np.array(features)


file_path = "../MEI/data/XU_pretrain_val_positive.fasta"
save_path = "feature"
os.makedirs(save_path, exist_ok=True)


sequences = read_file(file_path)
features = aac_feature(sequences)


output_df = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(features.shape[1])])
output_df.insert(0, 'Sequence', sequences)


fasta_filename = os.path.basename(file_path)
csv_filename = os.path.splitext(fasta_filename)[0] + '_aac_features.csv'
output_csv_path = os.path.join(save_path, csv_filename)

output_df.to_csv(output_csv_path, index=False)
print(f"{output_csv_path}")