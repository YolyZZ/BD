import pandas as pd
import numpy as np
from process import lstm_time_series
from sklearn.metrics import mean_squared_error

WEIGHT_TRANS_P = 4
WEIGHT_TRANS_N = 5
WEIGHT_TRANS_NE = 1

WEIGHT_HEAT = 1
WEIGHT_TRANSFORM = 2

WEIGHT_RADIO = 0.6
WEIGHT_COUNT = 0.1


def lstm_pre(filename):
    AirLine = lstm_time_series.Airline_Predict(filename)
    train_x, train_y, test_x, test_y, MMS = AirLine.load_data()

    predict_y, test_y = AirLine.train_model(train_x, train_y, test_x, test_y)

    # 对标注化后的数据还原
    predict_y = MMS.inverse_transform([[i] for i in predict_y])
    test_y = MMS.inverse_transform([[i] for i in test_y])
    # predict_y = lstm_time_series.eliminate_neg(predict_y)

    true_value = np.reshape(test_y, (test_y.size,))
    predict = np.reshape(predict_y, (predict_y.size,))
    mse = mean_squared_error(true_value, predict)
    # print('---------:MSE', mse)
    return predict


def predict_index(df_mass, df_media, event):

    # 自动concat 媒体+民众 'date', 'sentiment', 'link_keywords'
    # df1 = pd.read_csv("res/tw_res.csv", sep=',')
    # df2 = pd.read_csv("res/k_tw_res.csv", sep=',')
    # df1 = pd.read_csv("res/ct_res.csv", sep=',')
    # df2 = pd.read_csv("res/k_ct_res.csv", sep=',')
    df1 = df_mass
    df2 = df_media

    df1 = df1[['date', 'sentiment', 'link_keywords']]
    df2 = df2[['date', 'sentiment', 'link_keywords']]
    df = pd.concat([df1, df2], ignore_index=True)
    # df.to_csv("res/df_res.csv", encoding='utf_8', index=False, sep=',')

    link_count_ratio = (df[df['link_keywords'] != '[]'].shape[0]) / (len(df))

    df = df.drop(columns=['link_keywords'], axis=1)

    df['count'] = 1
    df['count'] = df.groupby(['date']).transform('count')

    df['positive'] = 0
    df['negative'] = 0
    df['neutral'] = 0
    df['positive'] = df[df['sentiment'] == 1].groupby(['date', 'sentiment']).transform('count')
    df['negative'] = df[df['sentiment'] == -1].groupby(['date', 'sentiment']).transform('count')
    df['neutral'] = df[df['sentiment'] == 0].groupby(['date', 'sentiment']).transform('count')

    df_count = df[['date', 'count', 'positive', 'negative', 'neutral']]
    df_count = df_count.fillna(0)
    df_count = df_count.drop_duplicates()

    df_count = df_count.groupby(['date', 'count']).sum() # sum up group by date&count
    df_count = df_count.reset_index()
    df_count['date'] = pd.to_datetime(df_count['date'])
    df_count = df_count.sort_values(by='date')  # sort by date

    # count_file_path = "res/total_count_res.csv"
    # df.to_csv(count_file_path, encoding='utf_8', index=False, sep=',')
    # lstm_pre(count_file_path)


    # 筛选近期的数据进行预测（固定值，后续要删除。取固定时间段内容输入，例如近一周，源头控制输入时间段）
    if event=="twdx":
        df_count = df_count[df_count['date'] >= '2021-04-01']  # filt date for tw
    elif event=="byct":
        df_count = df_count[df_count['date'] >= '2021-05-21']  # filt date for ct
    else:
        len_df = df_count.shape[0]
        df_count = df_count[len_df-50, len_df-1]


    # heat_percent
    df_count['heat_percent'] = ((df_count['count'] - df_count['count'].shift(1)) / df_count['count'].shift(1)) * 100
    # transform_percent
    df_count['transform_percent'] = ((WEIGHT_TRANS_P*abs(df_count['positive'] - df_count['positive'].shift(1))
                                     +WEIGHT_TRANS_N*abs(df_count['negative'] - df_count['negative'].shift(1))
                                      +WEIGHT_TRANS_NE*abs(df_count['neutral'] - df_count['neutral'].shift(1)))
                                     / (df_count['count'].shift(1)*(WEIGHT_TRANS_P+WEIGHT_TRANS_N+WEIGHT_TRANS_NE))) * 100
    # interference_percent
    df_count['normal_heat_percent'] = (df_count['heat_percent'] - df_count['heat_percent'][:].min()) / (
                df_count['heat_percent'][:].max() - df_count['heat_percent'][:].min()) * 100
    df_count['normal_transform_percent'] = (df_count['transform_percent'] - df_count['transform_percent'][:].min()) / (
                df_count['transform_percent'][:].max() - df_count['transform_percent'][:].min())*100
    df_count['interference_percent'] = (WEIGHT_HEAT*df_count['heat_percent'] + WEIGHT_TRANSFORM*df_count['transform_percent']) / (WEIGHT_HEAT + WEIGHT_TRANSFORM)
    df_count['normal_interference_percent'] = (df_count['interference_percent'] - df_count['interference_percent'][:].min() + 1) / (
                df_count['interference_percent'][:].max() - df_count['interference_percent'][:].min() + 100) * 100
    df_count['interference_effect'] = WEIGHT_RADIO*link_count_ratio*100 + WEIGHT_COUNT*df_count['normal_interference_percent'] + \
                                      WEIGHT_COUNT*df_count['normal_heat_percent'] + WEIGHT_COUNT*df_count['normal_transform_percent']

    # 三个指数的上升下降
    df_count['change_transform_percent'] = np.where(
        df_count['transform_percent'] > df_count['transform_percent'].shift(1), 1, 0)
    df_count['change_interference_percent'] = np.where(
        df_count['interference_percent'] > df_count['interference_percent'].shift(1), 1, 0)
    df_count['change_interference_effect'] = np.where(
        df_count['interference_effect'] > df_count['interference_effect'].shift(1), 1, 0)

    df_count = df_count.drop(columns=['normal_heat_percent','normal_transform_percent','interference_percent'], axis=1)

    df_count.to_csv("res/pre_tw_count_res.csv", encoding='utf_8', index=False, sep=',')
    # df_count.to_csv("res/pre_ct_count_res.csv", encoding='utf_8', index=False, sep=',')
    return df_count