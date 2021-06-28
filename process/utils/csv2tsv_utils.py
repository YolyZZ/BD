import pandas as pd


def csv2tsv(file_path):
    train_set = pd.read_csv("train.csv", sep=',', header=0)
    train_df_bert = pd.DataFrame({
        'label': train_set['result'],
        'text': train_set['text'].replace(r'\n', ' ', regex=True)
    })
    train_df_bert.to_csv('train.tsv', sep='\t', index=False, header=True)

