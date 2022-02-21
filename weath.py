import pandas as pd
import xgboost as xgb
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

data_path='F:\plan\seeds_dataset.txt'
data=pd.read_csv(data_path,header=None,sep='\s+',converters={7:lambda x:int(x)-1})
data.rename(columns={7:'lable'},inplace=True)
print(data)

# # # 生产一个随机数并选择小于0.8的数据
# mask=np.random.rand(len(data))<0.8
# train=data[mask]
# test=data[~mask]
#
# # 生产DMatrix
# xgb_train=xgb.DMatrix(train.iloc[:,:6],label=train.lable)
# xgb_test=xgb.DMatrix(test.iloc[:,:6],label=test.lable)



X=data.iloc[:,:6]
Y=data.iloc[:,7]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=100)
print(y_test)
xgb_train=xgb.DMatrix(X_train,label=y_train)
xgb_test=xgb.DMatrix(X_test,label=y_test)



# 设置模型参数

params={
    'objective':'multi:softmax',
    'eta':0.1,
    'max_depth':5,
    'num_class':3
}

watchlist=[(xgb_train,'train'),(xgb_test,'test')]
# 设置训练轮次，这里设置60轮
num_round=60
bst=xgb.train(params,xgb_train,num_round,watchlist)

# 模型预测

pred=bst.predict(xgb_test)


#模型评估

# error_rate=np.sum(pred!=test.lable)/test.lable.shape[0]
error_rate=np.sum(pred!=y_test)/y_test.shape[0]

print('测试集错误率(softmax):{}'.format(error_rate))

accuray=1-error_rate
print('测试集准确率：%.4f' %accuray)


# 模型保存
bst.save_model("F:\plan\modal002.model")


# 模型加载
bst=xgb.Booster()
bst.load_model("F:\plan\modal002.model")
pred=bst.predict(xgb_test)
