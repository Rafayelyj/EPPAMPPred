import pandas as pd


input_csv_path = "Progen2_train_dim.csv"
output_positive_path = "positive_Progen2_train.csv"
output_negative_path = "negative_Progen2_train.csv"



df = pd.read_csv(input_csv_path, header=0)



print(df.head())


positive_samples = df[df.iloc[:, -1] == 1].iloc[:, 1:-1]
negative_samples = df[df.iloc[:, -1] == 0].iloc[:, 1:-1]


positive_samples.to_csv(output_positive_path, index=False)
negative_samples.to_csv(output_negative_path, index=False)

