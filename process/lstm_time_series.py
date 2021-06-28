import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error


class Airline_Predict:
    def __init__(self, filename, sequence_length=10, split=0.8):
        self.filename = filename
        self.sequence_length = sequence_length
        self.split = split

    def load_data(self):
        df = pd.read_csv(self.filename, sep=',', usecols=[1])
        data_all = np.array(df).astype('float')

        # 数据归一化
        MMS = MinMaxScaler()
        data_all = MMS.fit_transform(data_all)

        # 构造输入lstm的3D数据
        data = []
        for i in range(len(data_all) - self.sequence_length - 1):
            data.append(data_all[i: i + self.sequence_length + 1])

        # global reshaped_data
        reshaped_data = np.array(data).astype('float64')

        # 打乱第一维数据
        # np.random.shuffle(reshaped_data)

        # 最后一列为数值标签
        x = reshaped_data[:, :-1]
        y = reshaped_data[:, -1]

        # 构建训练集
        split_boundary = int(reshaped_data.shape[0] * self.split)
        train_x = x[:split_boundary]

        # 构建测试集
        test_x = x[split_boundary:]
        # 训练集标签
        train_y = y[: split_boundary]
        # 测试集标签
        test_y = y[split_boundary:]

        return train_x, train_y, test_x, test_y, MMS


    def build_model(self):
        # LSTM函数的input_dim参数是输入的train_x的最后一个维度，最后一层采用了线性层
        model = Sequential()
        model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))

        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(output_dim=1))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def train_model(self, train_x, train_y, test_x, test_y):
        model = self.build_model()
        try:
            model.fit(train_x, train_y, batch_size=16, nb_epoch=100, validation_split=0.1)
            predict = model.predict(test_x)
            predict = np.reshape(predict, (predict.size,))  # 变成向量
            test_y = np.reshape(test_y, (test_y.size,))
        except KeyboardInterrupt:
            print('predict:', predict)
            print('test_y', test_y)

        # try:
        #     fig1 = plt.figure(1)
        #     plt.plot(predict, 'r')
        #     plt.plot(test_y, 'g-')
        #     plt.title('This pic is drawed using Standard Data')
        #     plt.legend(['predict', 'true'])
        #
        # except Exception as e:
        #     print(e)

        return predict, test_y

# def eliminate_neg(predict_y):
#     # 对负数的异常情况
#     for i in predict_y:
#         if i[0] < 0: i[0] = round(random.uniform(2, 99), 8)
#     return predict_y


if __name__ == '__main__':
    # test
    filename = 'res/fenqu_tw_count_res.csv'
    AirLine = Airline_Predict(filename)
    train_x, train_y, test_x, test_y, MMS = AirLine.load_data()

    predict_y, test_y = AirLine.train_model(train_x, train_y, test_x, test_y)

    # 对标注化后的数据还原
    predict_y = MMS.inverse_transform([[i] for i in predict_y])
    test_y = MMS.inverse_transform([[i] for i in test_y])
    # predict_y = eliminate_neg(predict_y)

    predict = np.reshape(predict_y, (predict_y.size,))
    true_value = np.reshape(test_y, (test_y.size,))
    print('---------Predict:', predict)
    print('---------True:', true_value)

    mse = mean_squared_error(true_value, predict)
    print('---------:MSE', mse)

    msle = mean_squared_log_error(true_value, predict)
    print('---------:MSLE', msle)

    fig2 = plt.figure(2)
    plt.plot(predict_y, 'g:', label='prediction')
    plt.plot(test_y, 'r-', label='True')
    plt.title('Drawed using Standard_Inversed Data')
    plt.legend(['predict', 'true'])
    plt.show()