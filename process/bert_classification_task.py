#! -*- coding:utf-8 -*-
# 情感分析，加载bert_zh权重

import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.layers import Lambda, Dense
from tqdm import tqdm
import os

from process.utils.file_utils import read_items_file, save_items_file, filename_add_suffix
from process.utils.model_utils import evaluate_statistic


set_gelu('tanh')  # 切换gelu版本
proj_path = os.path.abspath('.') # 单独跑训练时改成上级目录'..'

class BERTClassifier:
    def __init__(self, bert_model_dir, data_dir, model_path, num_classes, maxlen=128, batch_size=32):
        # Data And Model Directory
        self.data_dir = data_dir
        self.model_path = model_path

        # Classifier Class Num
        self.num_classes = num_classes

        # Model Config
        self.config_path = bert_model_dir + 'bert_config.json'
        self.checkpoint_path = bert_model_dir + 'bert_model.ckpt'
        self.dict_path = bert_model_dir + 'vocab.txt'

        # 建立分词器
        self.tokenizer = Tokenizer(self.dict_path, do_lower_case=True)
        self.maxlen = maxlen
        self.batch_size = batch_size

    def load_data(self, filename):
        """加载数据
        单条格式：(文本, 标签id)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                items = l.strip().split('\t')
                text, label = items[1], items[0]
                D.append((text, int(label)))
        return D

    def build_model(self):
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model='bert',
            return_keras_model=False,
        )

        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        output = Dense(
            units=self.num_classes,
            activation='softmax',
            kernel_initializer=bert.initializer
        )(output)

        model = keras.models.Model(bert.model.input, output)
        model.summary()

        return model

    def do_train(self):
        # 加载数据集
        train_data = self.load_data(self.data_dir + 'train.tsv')
        dev_data = self.load_data(self.data_dir + 'dev.tsv')
        test_data = self.load_data(self.data_dir + 'test.tsv')

        # 转换数据集
        train_generator = data_generator(train_data, self.batch_size, self.tokenizer, maxlen=self.maxlen)
        valid_generator = data_generator(dev_data, self.batch_size, self.tokenizer, maxlen=self.maxlen)
        test_generator = data_generator(test_data, self.batch_size, self.tokenizer, maxlen=self.maxlen)

        model = self.build_model()

        # 派生为带分段线性学习率的优化器。
        # 其中name参数可选，但最好填入，以区分不同的派生优化器。
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        model.compile(
            loss='sparse_categorical_crossentropy',
            # optimizer=Adam(1e-5),  # 用足够小的学习率
            optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
                1000: 1,
                2000: 0.1
            }),
            metrics=['accuracy'],
        )

        evaluator = Evaluator(model, self.model_path, valid_generator, test_generator)

        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=1,
            callbacks=[evaluator]
        )

    def do_eval(self):
        dev_data = self.load_data(self.data_dir + 'dev.tsv')
        valid_generator = data_generator(dev_data, self.batch_size, self.tokenizer)

        model = self.build_model()
        model.load_weights(self.model_path)

        print(u'final eval acc: %05f\n' % (evaluate(valid_generator, model)))

    def do_test(self):
        # 加载数据集
        test_data = self.load_data(self.data_dir + 'test.tsv')
        # 转换数据集
        test_generator = data_generator(test_data, self.batch_size, self.tokenizer)

        model = self.build_model()
        model.load_weights(self.model_path)

        y_pred_class_list = []
        y_true_class_list = []
        for x_true, y_true in tqdm(test_generator):
            y_pred_class = model.predict(x_true).argmax(axis=1)
            y_pred_class_list.extend(y_pred_class.tolist())
            y_true_class_list.extend(np.squeeze(y_true).tolist())

        evaluate_statistic(y_pred_class_list, y_true_class_list)
        # print(u'final test acc: %05f\n' % (evaluate(test_generator, model)))

    def do_predict(self, predict_file, res_file):
        # 加载数据集
        test_data = read_items_file(predict_file)
        test_label_data = [(items[0], '0') for items in test_data]
        # 转换数据集
        test_generator = data_generator(test_label_data, self.batch_size, self.tokenizer)

        model = self.build_model()
        model.load_weights(self.model_path)

        y_pred_list = np.zeros([1, self.num_classes])
        for x_true, _ in tqdm(test_generator):
            y_pred = model.predict(x_true)
            y_pred_list = np.row_stack((y_pred_list, y_pred))
        y_pred_list = y_pred_list[1:]

        prob_list = []
        class_list = []
        for i in range(len(test_data)):
            data = test_data[i]
            y_pred = y_pred_list[i].astype(np.str).tolist()
            y_class = y_pred_list[i].argmax(axis=0).astype(np.str)
            prob_list.append(data + y_pred)
            class_list.append(data + [y_class])

        class_res_file = filename_add_suffix(res_file, '_class')
        save_items_file(prob_list, res_file)
        save_items_file(class_list, class_res_file)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, data, batch_size, tokenizer, maxlen=128):
        DataGenerator.__init__(self, data, batch_size=batch_size)
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, model, model_path, valid_generator, test_generator):
        self.best_val_acc = 0.
        self.model = model
        self.model_path = model_path
        self.valid_generator = valid_generator
        self.test_generator = test_generator

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.valid_generator, self.model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.model_path)
        test_acc = evaluate(self.test_generator, self.model)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


def evaluate(data, model):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


# def build_sentiment_BERT_classifier():
#
#     bert_model_dir = '/Users/dongguangzhe/Documents/code/bert-master/model/chinese_L-12_H-768_A-12/'
#     # bert_model_dir = '/home/hadoop-ht-oedatamining/cephfs/data/dongguangzhe/bert-master/model/chinese_L-12_H-768_A-12/'
#
#     # data_dir = '../../hotel_data/sentiment_analysis/datasets_sample/'
#     data_dir = '../../hotel_data/sentiment_analysis/datasets/'
#     model_path = '../../model/hotel_sentiment_classification/hotel_sentiment_classification_model.ckpt'
#
#     batch_size = 32
#
#     sbc = HotelSentimentBERTClassifier(bert_model_dir, data_dir, model_path, batch_size=batch_size)
#     return sbc


def predict_contents_senti(test_file_name, res_file_path):
    # 预测情感值，不重新训练，直接初始化模型参数。
    # 传参待预测的文本文件名。民众媒体分开调用
    bert_model_dir = os.path.join(proj_path, 'chinese_L-12_H-768_A-12/')
    data_dir = os.path.join(proj_path, 'data_bert/')
    model_path = os.path.join(proj_path, 'saved_model/sentiment_classification_model.ckpt')

    # res_file = '../res_bert/sentiment_result.txt'
    res_file = res_file_path

    num_classes = 3
    maxlen = 256
    batch_size = 32

    bc = BERTClassifier(bert_model_dir, data_dir, model_path, num_classes, maxlen, batch_size)

    # bc.do_train()
    # bc.do_eval()
    # bc.do_predict(data_dir + 'mytest.tsv', res_file)
    bc.do_predict(data_dir + test_file_name, res_file)



if __name__ == '__main__':
    bert_model_dir = os.path.join(proj_path, 'chinese_L-12_H-768_A-12/')
    data_dir = os.path.join(proj_path, 'data_bert/')
    model_path = os.path.join(proj_path, 'saved_model/sentiment_classification_model.ckpt')

    res_file = os.path.join(proj_path, 'res_bert/sentiment_result.txt')

    num_classes = 3
    maxlen = 256
    batch_size = 32

    bc = BERTClassifier(bert_model_dir, data_dir, model_path, num_classes, maxlen, batch_size)

    bc.do_train()
    bc.do_eval()
    # bc.do_predict(data_dir + 'mytest.tsv', res_file)
