import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split


def train_valid_test_split(x_data, y_data, validation_size=0.1, test_size=0.1, shuffle=True):
    x_, x_test, y_, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle)
    valid_size = validation_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_, y_, test_size=valid_size, shuffle=shuffle)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == '__main__':
    # 带标注的数据，split训练集验证集测试集。训练阶段才需要，预测时没有调用此步
    pd_all_1 = pd.read_csv("../data_bert/label_mass_tw.csv")
    pd_all_2 = pd.read_csv("../data_bert/label_media_tw.csv")

    pd_all_1 = pd_all_1.rename(columns={'标题':'content', '情感属性':'sentiment'})
    pd_all_2 = pd_all_2.rename(columns={'摘要': 'content', '情感属性':'sentiment'})

    pd_all_1 = pd_all_1[['content', 'sentiment']]
    pd_all_2 = pd_all_2[['content', 'sentiment']]

    pd_all = pd.concat([pd_all_1, pd_all_2], ignore_index=True)
    # pd_all = pd_all_1

    # pd_all['content'] = pd_all['标题']
    pd_all['content'] = pd_all['content'].apply(lambda x: x.lstrip('\t'))
    pd_all['sentiment'].loc[pd_all['sentiment'] == '正面'] = 2
    pd_all['sentiment'].loc[pd_all['sentiment'] == '中性'] = 1
    pd_all['sentiment'].loc[pd_all['sentiment'] == '负面'] = 0
    pd_all['label'] = pd_all['sentiment']

    # for index, row in pd_all.iterrows():
    #     if len(row["content"]) > 200:
    #         print("-----------------", row["content"])
    #         pd_all.at[index, 'content'] = row["content"][0:200]
    #         print("-----------------after-----------", row["content"])

    # 文本截短
    pd_all['content'] = pd_all['content'].replace(r'\n', ' ', regex=True)\
        .apply(lambda x: x[0:128])

    x_data, y_data = pd_all['content'], pd_all['label']

    x_train, x_valid, x_test, y_train, y_valid, y_test = train_valid_test_split(x_data, y_data, 0.1, 0.1)

    train = pd.DataFrame({'label': y_train, 'x_train': x_train})
    train.to_csv("../data_bert/train.tsv", encoding='utf_8', index=False, sep='\t', header=False)
    valid = pd.DataFrame({'label': y_valid, 'x_valid': x_valid})
    valid.to_csv("../data_bert/dev.tsv", encoding='utf_8', index=False, sep='\t', header=False)
    test = pd.DataFrame({'label': y_test, 'x_test': x_test})
    test.to_csv("../data_bert/test.tsv", encoding='utf_8', index=False, sep='\t', header=False)

    # 服务器GPU版本的dataset格式，保留header：
    # train = pd.DataFrame({'label': y_train, 'x_train': x_train})
    # train.to_csv("D:/yoly/BD/data_gpu/train.csv", encoding='utf_8', index=False, sep='\t')
    # valid = pd.DataFrame({'label': y_valid, 'x_valid': x_valid})
    # valid.to_csv("D:/yoly/BD/data_gpu/dev.csv", encoding='utf_8', index=False, sep='\t')
    # test = pd.DataFrame({'label': y_test, 'x_test': x_test})
    # test.to_csv("D:/yoly/BD/data_gpu/test.csv", encoding='utf_8', index=False, sep='\t')