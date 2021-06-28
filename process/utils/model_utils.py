import numpy as np
from bert4keras.models import build_transformer_model, keras, Model
import pandas as pd

_NEG_INF = -1e9


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.

    Args:
      x: int tensor with any shape
      padding_value: int value that

    Returns:
      float tensor with same shape as x containing values 0 or 1.
        0 -> non-padding, 1 -> padding
    """
    return np.equal(x, padding_value)


def masked_sentence_avg_pooling_np(x, input_char, padding_value, cls_value=None, sep_value=None):
    '''
    :param x: [batch_size, input_length, hidden_size]
    :param input_char: [batch_size, input_length]
    ：param padding_value: int value that means padding
    :return: float tensor with shape [batch_size, input_length, hidden_size]
    '''
    padding = get_padding(input_char, padding_value)
    if cls_value is not None:
        cls_mask = get_padding(input_char, cls_value)
        padding = padding | cls_mask
    if sep_value is not None:
        sep_mask = get_padding(input_char, sep_value)
        padding = padding | sep_mask

    mask = np.expand_dims(1 - padding, axis=2)
    emb_mask = x * mask

    sum_h = np.sum(emb_mask, axis=1)
    sum_mask = np.sum(mask, axis=1)

    avg_emb = sum_h / sum_mask

    return avg_emb


def get_simcse_encoder(
        config_path,
        checkpoint_path,
        model='bert',
        pooling='first-last-avg',
        dropout_rate=0.1
):
    """建立编码器
    """
    assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']

    if pooling == 'pooler':
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model=model,
            with_pool='linear',
            dropout_rate=dropout_rate
        )
    else:
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model=model,
            dropout_rate=dropout_rate
        )

    outputs, count = [], 0
    while True:
        try:
            output = bert.get_layer(
                'Transformer-%d-FeedForward-Norm' % count
            ).output
            outputs.append(output)
            count += 1
        except:
            break

    if pooling == 'first-last-avg':
        outputs = [
            keras.layers.GlobalAveragePooling1D()(outputs[0]),
            keras.layers.GlobalAveragePooling1D()(outputs[-1])
        ]
        output = keras.layers.Average()(outputs)
    elif pooling == 'last-avg':
        output = keras.layers.GlobalAveragePooling1D()(outputs[-1])
    elif pooling == 'cls':
        output = keras.layers.Lambda(lambda x: x[:, 0])(outputs[-1])
    elif pooling == 'pooler':
        output = bert.output

    # 最后的编码器
    encoder = Model(bert.inputs, output)
    return encoder


def evaluate_statistic_dataframe(test_data):
    test_data['is_right'] = test_data['label'] == test_data['label_predict']

    predict_right = list(filter(lambda x: float(x) >= 0, test_data['label_predict'].where(test_data['is_right'])))
    accuracy = len(predict_right) / len(test_data)

    label_true = list(filter(lambda x: float(x) >= 0, test_data['label_predict'].where(test_data['label'] == '1')))
    recall = sum(np.array(label_true, dtype=np.int)) / len(label_true)

    predict_true = list(filter(lambda x: float(x) >= 0, test_data['label'].where(test_data['label_predict'] == '1')))
    precision = sum(np.array(predict_true, dtype=np.int)) / len(predict_true)

    f1_score = (2 * precision * recall) / (precision + recall)

    print('Accuracy:{}'.format(accuracy))
    print('Precision:{}'.format(precision))
    print('Recall:{}'.format(recall))
    print('F1_score:{}'.format(f1_score))


def evaluate_statistic(predict_label_list, label_list):
    test_data = pd.DataFrame(np.array([predict_label_list, label_list], dtype=np.str).transpose(),
                             columns=['label_predict', 'label'])
    evaluate_statistic_dataframe(test_data)
