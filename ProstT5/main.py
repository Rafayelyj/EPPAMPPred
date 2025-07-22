from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pandas as pd
from Bio import SeqIO
import os


local_model_path = 'Local'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


tokenizer = T5Tokenizer.from_pretrained(local_model_path, do_lower_case=False)


model = T5EncoderModel.from_pretrained(local_model_path)


model.to(device)


if device == 'cpu':
    model = model.float()
else:
    model = model.half()

def read_fasta(file_path):

    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq).upper())

    return sequences


def extract_features(sequences, batch_size=1):
    features = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]

        batch_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch_sequences]

        batch_sequences = ["<AA2fold> " + s for s in batch_sequences]

        inputs = tokenizer.batch_encode_plus(batch_sequences, add_special_tokens=True, padding="longest", return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding_repr = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

        batch_features = embedding_repr.last_hidden_state[:, 0, :].detach().cpu().numpy()
        features.extend(batch_features)

    return features

def save_features_to_csv(features, file_path):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = pd.DataFrame(features)
    df.to_csv(file_path, index=False)


def main():
    fasta_file_path = "data\XU_pretrain_train_positive_example.fasta"
    output_csv_path = "feature\XU_pretrain_train_positive_example.csv"


    sequences = read_fasta(fasta_file_path)


    features = extract_features(sequences, batch_size=1)


    save_features_to_csv(features, output_csv_path)


if __name__ == "__main__":
    main()