import os

from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
import torch
import pandas as pd
from Bio import SeqIO


model_path = "Progen2_small_local"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = Tokenizer.from_file(f"{model_path}/tokenizer.json")




fasta_file_path = "data/XU_pretrain_val_negative.fasta"

sequences = []
ids = []

for record in SeqIO.parse(fasta_file_path, "fasta"):
    sequences.append(str(record.seq))
    ids.append(record.id)


input_ids = [torch.tensor(tokenizer.encode(seq).ids).unsqueeze(0).to(model.device) for seq in sequences]


features = []
model.eval()
with torch.no_grad():
    for i, ids_tensor in enumerate(input_ids):
        outputs = model(ids_tensor, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]

        feature_vector = hidden_states[0, 0, :].cpu().numpy()
        features.append(feature_vector)


output_csv_path = "feature/XU_pretrain_val_negative_features.csv"
output_dir = os.path.dirname(output_csv_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.DataFrame(features, index=ids)
df.to_csv(output_csv_path, index_label="ID")



