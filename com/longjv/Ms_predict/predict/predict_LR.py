import numpy as np
import pandas as pd
df=pd.read_csv('C:\\Users\\duozhun\\Desktop\\日消数据\\测试数据\\re_item_id_ds_1.csv')
df=df.dropna()
# print(df.describe())


def accuracy(y_pred, y_test, res):
  num_state_1,num_state_0=0,0
  y_pred=y_pred.tolist()
  y_test = y_test.tolist()
  res= res.tolist()
  for i in range(len(y_pred)):
   if y_test[i] <= 2:
    if abs(res[i]) < 2:
     num_state_1=num_state_1+1
    else:num_state_0=num_state_0+1
   elif y_test[i] > 2 and y_test[i] <= 10:
    if abs(res[i]) < 3:
     num_state_1 = num_state_1 + 1
    else:
     num_state_0 = num_state_0 + 1
   elif y_test[i] > 10 and y_test[i] <= 20:
    if abs(res[i]) < y_test[i] * 0.2:
     num_state_1 = num_state_1 + 1
    else:
     num_state_0 = num_state_0 + 1
   elif y_test[i] > 20 and y_test[i] <= 100:
    if abs(res[i]) < y_test[i] * 0.15:
     num_state_1 = num_state_1 + 1
    else:num_state_0 = num_state_0 + 1
   elif y_test[i] > 100 :
     if abs(res[i]) < y_test[i] * 0.05:
      num_state_1 = num_state_1 + 1
     else:
      num_state_0 = num_state_0 + 1
  accuracy =num_state_1 / len(y_pred)
  print("accuracy=",'%.4f'%accuracy)

featue_base=df[['dm_1','dm_2','dm_3','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']]

df['dm_mean']=featue_base.mean(axis=1)

feature_cols =['dm_1','dm_2','dm_3','dm_4','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']
# feature_cols =['dm_1','dm_2','dm_3','dm_4','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']
X=df[['dm_mean','dm_1','dm_2','dm_3','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']]
# X=df[['dm_1','dm_2','dm_3','dm_4','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']]
y=df['ds']
# print(X)
# print(y)

#对特征进行归一化
from sklearn.preprocessing import Normalizer
X=Normalizer().fit_transform(X.astype(float))


# #对特征进行对数转化
# from numpy import log1p
# from sklearn.preprocessing import FunctionTransformer
# X=FunctionTransformer(log1p).fit_transform(X)

from sklearn.model_selection  import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=1)
# print (X_train.shape)
# print (y_train.shape)
# print (X_test.shape)
# print (y_test.shape)


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

model=linreg.fit(X_train, y_train)
print (model )
# print (linreg.intercept_)
# print (linreg.coef_)
# zip(feature_cols, linreg.coef_)

train_score = model.score(X_train, y_train)
print(train_score)
test_score=model.score(X_test, y_test)
print(test_score)

y_pred = linreg.predict(X_test)

#计算RMSE

from sklearn import metrics
res=y_pred-y_test
print ("RMSE by hand:",'%.2f'%np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
accuracy(y_pred, y_test, res)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
# plt.plot(range(len(y_pred)),y_test,'r',label="test")
# plt.legend(loc="upper right") #显示图中的标签
# plt.xlabel("the number of sales")
# plt.ylabel('value of sales')
# plt.show()