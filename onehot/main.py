import os
import pandas as pd
import numpy as np
from Bio import SeqIO


def read_file(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq).upper())
    return sequences


def onehot_feature(sequences, feature_dim=300):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

    features = []
    for seq in sequences:

        onehot = np.zeros((len(seq), len(amino_acids)))


        for i, aa in enumerate(seq):
            if aa in aa_to_index:
                onehot[i, aa_to_index[aa]] = 1

        seq_feature = onehot.mean(axis=0)
        features.append(seq_feature)

    return np.array(features)



file_path = "data/XU_pretrain_val_positive.fasta"
save_path = "feature"
os.makedirs(save_path, exist_ok=True)


sequences = read_file(file_path)
features = onehot_feature(sequences)


output_df = pd.DataFrame(features, columns=[f'Feature_{i + 1}' for i in range(features.shape[1])])
output_df.insert(0, 'Sequence', sequences)


fasta_filename = os.path.basename(file_path)
csv_filename = os.path.splitext(fasta_filename)[0] + '_features.csv'
output_csv_path = os.path.join(save_path, csv_filename)

output_df.to_csv(output_csv_path, index=False)
