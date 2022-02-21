from numpy import loadtxt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

# {'colsample_bytree': 0.8, 'gamma': 0.7, 'max_depth': 7, 'min_child_weight': 6, 'subsample': 0.7}

modal = XGBClassifier(
    seed=4,  # 随机种子 复现
    num_class=11,
    booster='gbtree',
    # scale_pos_weight=1,  # 正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时scale_pos_weight=10。
    # booster='gblinear',
    nthread=-1,  # nthread=-1时，使用全部CPU进行并行运算（默认）nthread=1时，使用1个CPU进行运算。
    silent=1,  # silent=0时，不输出中间过程（默认）silent=1时，输出中间过程
    subsample=0.8,  # 使用的数据占全部训练集的比例。防止overfitting。默认值为1，典型值为0.5-1。
    colsample_bytree=0.8,  # 使用的特征占全部特征的比例。防止overfitting。默认值为1，典型值为0.5-1。
    learning_rate=0.08,  # 学习率，控制每次迭代更新权重时的步长，值越小，训练越慢。默认0.3，典型值为0.01-0.2。
    n_estimators=1000,  # 总共迭代的次数，即决策树的个数，数值大没关系，cv会自动返回合适的n_estimators
    max_depth=7,  # 树的深度，默认值为6，典型值3-10。
    min_child_weight=3,  # 值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）。默认值为1
    gamma=0,  # 惩罚项系数，指定节点分裂所需的最小损失函数下降值。
    gpu_id=0,
    tree_method='gpu_hist',
    # objective='binary:logistic',
    # objective='reg:linear',
    alpha=1e-5,
    eval_metric='merror',
    objective='multi:softprob'
)


#  'alpha':[1e-5, 1e-2, 0.1, 1, 100,0.001,0.005,0.01,0.05]
# 1.0962\0.2822 1.2228 0.2819 \ 1.2716 0.2830 \ 1.3413 0.2832  \ 1.5277 0.2775 \ 1.3698 0.2806 \1.1425 0.2830 \
def modelfit(alg, train_X, train_y, test_X, test_y, useTrainCV=True, cv_folds=None, early_stopping_rounds=100):
    list = []
    if useTrainCV:
        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(train_X, label=train_y)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds=cv_folds,
                          metrics='merror', early_stopping_rounds=early_stopping_rounds)

        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators=n_estimators)
        print(n_estimators)
        print(cvresult)
    # Fit the algorithm on the data
    # print("train_y",train_y)
    alg.fit(train_X, train_y)
    # 打印权重
    # for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    #     print('%s: ' % importance_type, alg.get_booster().get_score(importance_type=importance_type))
    # Predict training set:
    train_predprob = alg.predict_proba(train_X)

    logloss = metrics.log_loss(train_y, train_predprob)

    # Print model report:
    print("logloss of train :%.4f" % logloss)
    y_pred = np.array(alg.predict(test_X))

    predictions = [round(value) for value in y_pred]
    # print('AUC: %.4f' % metrics.roc_auc_score(test_y, y_pred))
    print('ACC: %.4f' % metrics.accuracy_score(test_y, predictions))
    # print('Recall: %.4f' % metrics.recall_score(test_y, predictions))
    # print('Precesion: %.4f' % metrics.precision_score(test_y, predictions))
    # print('F1-score: %.4f' % metrics.f1_score(test_y, predictions))
    # print('test_y',test_y)
    # print('y_pred',y_pred)
    proba = alg.predict_proba(test_X)
    predict = alg.predict(test_X)
    # print(len(test_X))
    count = 0
    sum = 0
    for i in range(len(proba)):
        sum += 1
        if predict[i] == test_y[i]:
            count += 1
        if predict[i] == 1:
            stock = {'code': str(target[i][0]), 'proba': proba[i][0], 'type': predict[i]}
            # stock['income'] = income[i]
            list.append(stock)
    print("成功率：", count / sum)
    list.sort(key=lambda x: x['proba'], reverse=True)
    # print(proba) #打印概率
    return list
    # print(alg.predict(test_X)) #打印标签


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

result = ""

# for j in range(1):
#     print(j)
# 准备数据
dataX = loadtxt('E:\\plan\\xgboost\\resources\\2004_trans.csv', delimiter=",")
train_X = dataX[:, 1:len(dataX[0]) - 1]
train_y = dataX[:, len(dataX[0]) - 1]
# X = dataX[:,1:len(dataX[0])-2]
# Y = dataX[:,len(dataX[0])-1]

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=100)


dataY = loadtxt('E:\\plan\\xgboost\\resources\\2004_tests.csv', delimiter=",")

test_X = dataY[:, 1:len(dataY[0]) - 1]
test_y = dataY[:, len(dataY[0]) - 1]
target = dataY[:, 0:1]

list = modelfit(modal, train_X, train_y, test_X, test_y, cv_folds=kfold)
for i in range((len(list))):
    print(list[i]['code'], list[i]['type'], list[i]['proba'])

# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print(y_pred)

print(result)
