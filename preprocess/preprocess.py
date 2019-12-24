#! -*- coding:utf-8 -*-
import json
import numpy as np
import pandas as pd  # ! -*- coding:utf-8 -*-
import json
import numpy as np
import pandas as pd
import time
import random

l = np.array([['1', '1', '1', '1'],
              ['1', '0', '2', '2'],
              ['1', '2', '3', '3'],
              ['1', '2', '3', '4']])
k = np.array([[55, 55, 55, 55],
              [55,  0,  6,  6],
              [55,  6,  8,  9],
              [55,  6,  9,  2]])

random.seed(123)

train_interrelation = pd.read_csv('./input/Train_Interrelation.csv', dtype=str)
print("Train_Interrelation", len(train_interrelation))

sum = [0 for i in range(5)]
for i in train_interrelation['Level']:
    sum[int(i)] += 1
for i in range(1, 5):
    print('等级', i, sum[i])

tur = set()
for i in train_interrelation['Rid']:
    tur.add(i)
print(len(tur))
train_re = train_interrelation.sort_values(['Rid'])
sum = 0
for i in range(len(train_re)):
    j = 1
    flag = 0
    while i + j < len(train_re) and train_re['Rid'].values[i] == train_re['Rid'].values[i + j]:
        flag = 1
        if flag == 1 and train_re['Rid'].values[i] != train_re['Rid'].values[i + j]:
            break
        c = int(train_re['Level'].values[i]) - 1
        r = int(train_re['Level'].values[i + j]) - 1
        if l[c][r] != '0' and random.randint(0, k[c][r]) == 0:
            sum += 1
            # print(train_re['Rid'].values[i], train_re['Aid'].values[i], train_re['Aid'].values[i + j])
            train_interrelation.loc[i * 1000000 + j] = \
                [train_re['Rid'].values[i],
                 train_re['Aid'].values[i],
                 train_re['Aid'].values[i + j],
                 l[int(train_re['Level'].values[i]) - 1][int(train_re['Level'].values[i + j]) - 1]]
        j += 1

train_interrelation.to_csv('./Train_Interrelation.csv', index=False)

print("Train_Interrelation", len(train_interrelation))
print('sum', sum)

sum = [0 for i in range(5)]
for i in train_interrelation['Level']:
    sum[int(i)] += 1
for i in range(1, 5):
    print('等级', i, sum[i])


