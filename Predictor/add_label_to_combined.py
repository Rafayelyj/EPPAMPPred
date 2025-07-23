import sys
import pandas as pd

def add_label_to_combined(combined_file_path):
    if 'positive' in combined_file_path:
        label_value = 1
    elif 'negative' in combined_file_path:
        label_value = 0
    else:
        print("The filename does not contain 'positive' or 'negative', label column not added.")
        return
    df = pd.read_csv(combined_file_path, index_col="ID")
    df['label'] = label_value
    df.to_csv(combined_file_path, index_label="ID")
    print(f"Label column added to {combined_file_path}, value is {label_value}")

if __name__ == "__main__":
    add_label_to_combined(sys.argv[1]) 