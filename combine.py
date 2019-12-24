import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--k", default=6, type=int, required=False)
parser.add_argument("--a", default=0.52, type=float, required=False)
parser.add_argument("--op", default=0, type=int, required=False)
parser.add_argument("--n", default=0, type=int, required=False)
args = parser.parse_args()
k = args.k
a = args.a
op = args.op
n = args.n

test = pd.read_csv('./input/TestPrediction.csv')
oof_test = np.loadtxt('./submit/w_0.txt')
print(oof_test)
print('-----------------------')
oof_test1 = np.loadtxt('./submit/w_{}.txt'.format(op))
oof_test += oof_test1 * n
print(oof_test)
print('-----------------------')

for i in range(1, k):
    oof_test1 = np.loadtxt('./submit/w_{}.txt'.format(i))
    oof_test += oof_test1

print(oof_test)
print('-----------------------')
for i in range(len(oof_test)):
    oof_test[i][0] = oof_test[i][0] * (1 - a)
    oof_test[i][1] = oof_test[i][1] * a
    oof_test[i][2] = oof_test[i][2] * a
    oof_test[i][3] = oof_test[i][3] * (1 - a)
print(oof_test)
test['Level'] = np.argmax(oof_test, axis=1) + 1
test[['Guid', 'Level']].to_csv('./submit/submit.csv', index=False)
