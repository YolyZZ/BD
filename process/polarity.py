import os
import pandas as pd

OUTPUT_FILE="output_from_gpu"

if __name__ == '__main__':
    pd_all = pd.read_csv("data_bert/test_results.tsv", sep='\t', header=None)
    data = pd.DataFrame(columns=['polarity'])
    print(pd_all.shape)

    for index in pd_all.index:
        positive_score = pd_all.loc[index].values[0]
        neutral_score = pd_all.loc[index].values[1]
        negative_score = pd_all.loc[index].values[2]

        print(positive_score, neutral_score, negative_score)

        if max(positive_score, neutral_score, negative_score) == positive_score:
            data.loc[index + 1] = "1"
        elif max(positive_score, neutral_score, negative_score) == neutral_score:
            data.loc[index + 1] = "0"
        elif max(positive_score, neutral_score, negative_score) == negative_score:
            data.loc[index + 1] = "-1"

    # data.to_csv(os.path.join("res/pred_result.txt"), sep='\t', index=False)
    data.to_csv("res/pred_senti_res.csv", encoding='utf_8', index=False, sep='\t')
