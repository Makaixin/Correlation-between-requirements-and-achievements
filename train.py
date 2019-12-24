#! -*- coding:utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
import tensorflow as tf
import os
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
import random
import argparse

# 超参数
parser = argparse.ArgumentParser()
parser.add_argument("--counter", default=0, type=int, required=False)
parser.add_argument("--name", default='', type=str, required=False)
parser.add_argument("--model", default=0, type=int, required=False)
parser.add_argument("--model1", default=1, type=int, required=False)
parser.add_argument("--title_len", default=128, type=int, required=False)
parser.add_argument("--content_len", default=512, type=int, required=False)
parser.add_argument("--learning_rate", default=5e-5, type=float, required=False)
parser.add_argument("--min_learning_rate", default=1e-5, type=float, required=False)
parser.add_argument("--random_seed", default=123, type=int, required=False)
parser.add_argument("--batch_size", default=16, type=int, required=False)
parser.add_argument("--epoch", default=8, type=int, required=False)
parser.add_argument("--fold", default=7, type=int, required=False)

args = parser.parse_args()
counter = args.counter
name = args.name
model = args.model
model1 = args.model1
MAX_LENT = args.title_len
MAX_LENC = args.content_len
learning_rate = args.learning_rate
min_learning_rate = args.min_learning_rate
random_seed = args.random_seed
bs = args.batch_size
epoch = args.epoch
fold = args.fold

# cpu运行
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 预训练所在文件夹
bert_path = ['chinese_L-12_H-768_A-12',
             'chinese_wwm_ext_L-12_H-768_A-12',
             'chinese_roberta_wwm_ext_L-12_H-768_A-12',
             'roeberta_zh_L-24_H-1024_A-16']

# 不全部占满显存, 按需分配
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

# 加载对应预训练
config_path = './ckpt/' + bert_path[model] + '/bert_config.json'
checkpoint_path = './ckpt/' + bert_path[model] + '/bert_model.ckpt'
dict_path = './ckpt/' + bert_path[model] + '/vocab.txt'
config_path1 = './ckpt/' + bert_path[model1] + '/bert_config.json'
checkpoint_path1 = './ckpt/' + bert_path[model1] + '/bert_model.ckpt'
dict_path1 = './ckpt/' + bert_path[model1] + '/vocab.txt'

# 加载词汇表
token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)
token_dict1 = {}
with open(dict_path1, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict1[token] = len(token_dict1)
tokenizer1 = Tokenizer(token_dict1)

file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp + '.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)


# 数据读入与预处理
def read_data(file_path, id, name):
    train_id = []
    train_title = []
    train_text = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for idx, line in enumerate(f):
            line = line.strip().split(',')
            train_id.append(line[0].replace('\'', '').replace(' ', ''))
            train_title.append(line[1])
            train_text.append('，'.join(line[2:]))
    output = pd.DataFrame(dtype=str)
    output[id] = train_id
    output[name + '_title'] = train_title
    output[name + '_content'] = train_text
    return output


# 读取数据
train_interrelation = pd.read_csv('./input/Train_Interrelation.csv', dtype=str)
Train_Achievements = read_data('./input/Train_Achievements.csv', 'Aid', 'Achievements')
Requirements = read_data('./input/Requirements.csv', 'Rid', 'Requirements')
TestPrediction = pd.read_csv('./input/TestPrediction.csv', dtype=str)
Test_Achievements = read_data('./input/Test_Achievements.csv', 'Aid', 'Achievements')

# 将train和test数据表连接成大表并选出有用信息
train = pd.merge(train_interrelation, Train_Achievements, on='Aid', how='left')
train = pd.merge(train, Requirements, on='Rid', how='left')
test = pd.merge(TestPrediction, Test_Achievements, on='Aid', how='left')
test = pd.merge(test, Requirements, on='Rid', how='left')


# 对数据进行预处理。
# 将内容为空白或如“图片”之类的无用信息替换为对应标题
for i in range(len(train)):
    if len(train['Achievements_content'][i]) < 14:
        train['Achievements_content'][i] = train['Achievements_title'][i]
    if len(train['Requirements_content'][i]) < 10:
        train['Requirements_content'][i] = train['Requirements_title'][i]
print("train预处理完毕")

for i in range(len(test)):
    if len(test['Achievements_content'][i]) < 14:
        test['Achievements_content'][i] = test['Achievements_title'][i]
    if len(test['Requirements_content'][i]) < 10:
        test['Requirements_content'][i] = test['Requirements_title'][i]
print("test预处理完毕")

train_achievements = train['Achievements_title'].values
train_requirements = train['Requirements_title'].values
train_achievementsc = train['Achievements_content'].values
train_requirementsc = train['Requirements_content'].values

test_achievements = test['Achievements_title'].values
test_requirements = test['Requirements_title'].values
test_achievementsc = test['Achievements_content'].values
test_requirementsc = test['Requirements_content'].values

labels = train['Level'].astype(int).values - 1
labels_cat = to_categorical(labels)
labels_cat = labels_cat.astype(np.int32)


# 数据生成
class data_generator:
    def __init__(self, data, batch_size=bs):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, X2, X3, X4, y = self.data
            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_, Y = [], [], []
            X, X_, Z = [], [], []
            for c, i in enumerate(idxs):
                achievements = X1[i]
                requirements = X2[i]
                achievementsc = X3[i]
                requirementsc = X4[i]
                t, t_ = tokenizer.encode(first=achievements, second=requirements, max_len=MAX_LENT)
                x, x_ = tokenizer1.encode(first=achievementsc, second=requirementsc, max_len=MAX_LENC)
                T.append(t)
                T_.append(t_)
                X.append(x)
                X_.append(x_)
                Y.append(y[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = np.array(T)
                    T_ = np.array(T_)
                    X = np.array(X)
                    X_ = np.array(X_)
                    Y = np.array(Y)
                    yield [T, T_, X, X_], Y
                    T, T_, Y = [], [], []
                    X, X_, Z = [], [], []


# 模型构建
# 在第一个bert中对标题进行相似度判别
# 在第二个bert中对内容进行相似度判别
# 分别取出[CLS]进行拼接后进行分类
def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    bert_model1 = load_trained_model_from_checkpoint(config_path1, checkpoint_path1)
    for l in bert_model.layers:
        l.trainable = True

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))
    X1 = Input(shape=(None,))
    X2 = Input(shape=(None,))

    T = bert_model([T1, T2])
    X = bert_model1([X1, X2])

    T = Lambda(lambda x: x[:, 0])(T)
    X = Lambda(lambda x: x[:, 0])(X)

    T = Concatenate(axis=-1)([T, X])
    T = Dense(384)(T)
    # T = Dropout(0.1)(T)
    output = Dense(4, activation='softmax')(T)

    model = Model([T1, T2, X1, X2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['MAE']
    )
    model.summary()
    return model


class Evaluate(Callback):
    def __init__(self, val_data, val_index):
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0

    # 第一个epoch用来warmup，第二个epoch把学习率降到最低
    def on_batch_begin(self, batch, logs=None):
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        score, acc, f1 = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
            model.save_weights('./model_save/bert{}.w'.format(fold))
        else:
            self.early_stopping += 1
        logger.info('fold: %d, lr: %.6f, score: %.4f, acc: %.4f, f1: %.4f,best: %.4f\n' % (
            fold, self.lr, score, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_x2, val_x3, val_x4, val_y, val_cat = self.val_data

        for i in tqdm(range(len(val_x1))):
            achievements = val_x1[i]
            requirements = val_x2[i]
            achievementsc = val_x3[i]
            requirementsc = val_x4[i]

            t1, t1_ = tokenizer.encode(first=achievements, second=requirements, max_len=MAX_LENT)
            x1, x1_ = tokenizer1.encode(first=achievementsc, second=requirementsc, max_len=MAX_LENC)

            T1, T1_ = np.array([t1]), np.array([t1_])
            X1, X1_ = np.array([x1]), np.array([x1_])

            _prob = model.predict([T1, T1_, X1, X1_])

            oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(np.argmax(_prob, axis=1)[0] + 1)
            prob.append(_prob[0])

        score = 1.0 / (1 + mean_absolute_error(val_y + 1, self.predict))
        acc = accuracy_score(val_y + 1, self.predict)
        f1 = f1_score(val_y + 1, self.predict, average='macro')
        return score, acc, f1


skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_seed)


def predict(data):
    prob = []
    val_x1, val_x2, val_x3, val_x4 = data
    for i in tqdm(range(len(val_x1))):
        achievements = val_x1[i]
        requirements = val_x2[i]
        achievementsc = val_x3[i]
        requirementsc = val_x4[i]

        t1, t1_ = tokenizer.encode(first=achievements, second=requirements, max_len=MAX_LENT)
        x1, x1_ = tokenizer1.encode(first=achievementsc, second=requirementsc, max_len=MAX_LENC)

        T1, T1_ = np.array([t1]), np.array([t1_])
        X1, X1_ = np.array([x1]), np.array([x1_])

        _prob = model.predict([T1, T1_, X1, X1_])
        prob.append(_prob[0])
    return prob


oof_train = np.zeros((len(train), 4), dtype=np.float32)
oof_test = np.zeros((len(test), 4), dtype=np.float32)
logger.info("加载{}和{}".format(bert_path[model], bert_path[model1]))

timestamp = time.time()

for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements, labels)):
    logger.info('------------ %d fold take: %.1f minute ------------' % (fold, (time.time() - timestamp) / 60))
    timestamp = time.time()
    x1 = train_achievements[train_index]
    x2 = train_requirements[train_index]
    x3 = train_achievementsc[train_index]
    x4 = train_requirementsc[train_index]
    y = labels_cat[train_index]

    val_x1 = train_achievements[valid_index]
    val_x2 = train_requirements[valid_index]
    val_x3 = train_achievementsc[valid_index]
    val_x4 = train_requirementsc[valid_index]
    val_y = labels[valid_index]
    val_cat = labels_cat[valid_index]

    train_D = data_generator([x1, x2, x3, x4, y])
    evaluator = Evaluate([val_x1, val_x2, val_x3, val_x4, val_y, val_cat], valid_index)

    model = get_model()
    model.fit_generator(train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=epoch,
                        callbacks=[evaluator]
                        )
    model.load_weights('./model_save/bert{}.w'.format(fold))
    oof_test += predict([test_achievements, test_requirements, test_achievementsc, test_requirementsc])
    K.clear_session()

oof_test /= epoch

cv_score = 1.0 / (1 + mean_absolute_error(labels + 1, np.argmax(oof_train, axis=1) + 1))
logger.info(cv_score)

np.savetxt('./submit/w_{}.txt'.format(counter), oof_test)
test['Level'] = np.argmax(oof_test, axis=1) + 1
test[['Guid', 'Level']].to_csv('./submit/{}.csv'.format(counter), index=False)
