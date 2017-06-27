import numpy as np
import pandas as pd
df_1=pd.read_csv('C:\\Users\\duozhun\\Desktop\\日消数据\\测试数据\\re_item_id_ds_1.csv')
#使用总样本集
df_sum=pd.read_csv('C:\\Users\\duozhun\\Desktop\\日消数据\\测试数据\\re_item_id_ds.csv')
from sklearn.model_selection  import train_test_split
df,df_else = train_test_split(df_sum,test_size=0.9)
df=df.dropna()
# lrr=lrr[lrr.dm_1 > 0]
# print(df.describe())

def accuracy(y_pred, y_test, res,x_i):
  num_state_1,num_state_0=0,0
  y_pred=y_pred.tolist()
  y_test = y_test.tolist()
  res= res.tolist()
  for i in range(len(y_pred)):
   if y_test[i] <= 2:
    if abs(res[i]-1) < 2:
     num_state_1=num_state_1+1
    else:num_state_0=num_state_0+1
   elif y_test[i] > 2 and y_test[i] <= 10:
    if abs(res[i]) < 4:
     num_state_1 = num_state_1 + 1
    else:
     num_state_0 = num_state_0 + 1
   elif y_test[i] > 10 and y_test[i] <= 20:

    if abs(res[i]+2) < 6:
     num_state_1 = num_state_1 + 1
    else:
     num_state_0 = num_state_0 + 1
   elif y_test[i] > 20 and y_test[i] <= 100:
    if abs(res[i]) < y_test[i] * 0.2:
     num_state_1 = num_state_1 + 1
    else:num_state_0 = num_state_0 + 1
   elif y_test[i] > 100 :
     if abs(res[i]) < y_test[i] * 0.05:
      num_state_1 = num_state_1 + 1
     else:
      num_state_0 = num_state_0 + 1
  accu =num_state_1 / len(y_pred)
  x_dict[x_i] = '%.2f%%'%(accu*100)
  print("accuracy =",'%.2f%%'%(accu*100))


X_list,y_list=[], []

def feature_deal(df):
    # 特征处理
    featue_base = df[
        ['dm_1', 'dm_2', 'dm_3', 'dm_4', 'dm_5', 'dm_6', 'dm_7', 'dm_8', 'dm_9', 'dm_10', 'dm_11', 'dm_12', 'dm_13',
         'dm_14', 'dm_15', 'dm_16', 'dm_17', 'dm_18', 'dm_19', 'dm_20', 'dm_21', 'dm_22', 'dm_23', 'dm_24', 'dm_25',
         'dm_26', 'dm_27', 'dm_28', 'dm_29', 'dm_30']]

    df['dm_mean'] = featue_base.mean(axis=1)
    # 特征一阶差分,且小于0的数全部定义为0
    featue_base_diff = featue_base.diff(periods=1, axis=1).drop('dm_1', 1)
    featue_base_diff = featue_base_diff.where(featue_base_diff > 0, 0)
    for i in featue_base_diff:
        featue_base_diff.rename(columns={i: i + '_diff'}, inplace=True)
    # 添加星期特征和月份
    import time, datetime
    from datetime import datetime
    df['weekday'] = df['dt'].apply(
        lambda x: datetime.strptime(datetime.strptime(x, "%Y/%m/%d").strftime("%Y%m%d"), '%Y%m%d').weekday())
    df['month'] = df['dt'].apply(
        lambda x: datetime.strptime(datetime.strptime(x, "%Y/%m/%d").strftime("%Y%m%d"), '%Y%m%d').month)
    # 对星期和月份特征进行one_hot编码,转化成多维变量
    month_dummies = pd.get_dummies(df['month'])
    weekday_dummies = pd.get_dummies(df['weekday'])
    monthname_list = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    weekdayname_list = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期天']
    for i in month_dummies:
        month_dummies.rename(columns={i: monthname_list[int(i - 1)]}, inplace=True)
    for i in weekday_dummies:
        weekday_dummies.rename(columns={i: weekdayname_list[int(i - 1)]}, inplace=True)
    feature_cols = ['dm_1', 'dm_2', 'dm_3', 'dm_4', 'dm_5', 'dm_6', 'dm_7', 'dm_8', 'dm_9', 'dm_10', 'dm_11', 'dm_12',
                    'dm_13', 'dm_14', 'dm_15', 'dm_16', 'dm_17', 'dm_18', 'dm_19', 'dm_20', 'dm_21', 'dm_22', 'dm_23',
                    'dm_24', 'dm_25', 'dm_26', 'dm_27', 'dm_28', 'dm_29', 'dm_30']
    # feature_cols =['dm_1','dm_2','dm_3','dm_4','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']
    X = df[['weekday', 'dm_mean', 'dm_1', 'dm_2', 'dm_4', 'dm_3', 'dm_6', 'dm_7', 'dm_8', 'dm_9', 'dm_10', 'dm_11',
            'dm_12', 'dm_13', 'dm_14', 'dm_15', 'dm_17', 'dm_18', 'dm_19', 'dm_20', 'dm_21', 'dm_22', 'dm_23', 'dm_24',
            'dm_25', 'dm_26', 'dm_27', 'dm_29']]
    # X=df[['dm_1','dm_2','dm_3','dm_4','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']]
    y = df['ds']
    X = X.join(featue_base_diff[['dm_2_diff', 'dm_3_diff', 'dm_6_diff']])


    # X=X.join(month_dummies).join(weekday_dummies)
    # X.join(df['weekday'])
    # print(X)


    ##添加对数化特征
    # from numpy import log1p
    # from sklearn.preprocessing import FunctionTransformer
    #
    # log_feature=FunctionTransformer(log1p).fit_transform(featue_base)
    # df_log = pd.DataFrame(log_feature)
    # X= X.join(df_log)
    X_list.append(X)
    y_list.append(y)


from sklearn.model_selection import train_test_split
feature_deal(df)
feature_deal(df_1)
X_train, X_test, y_train, y_test = train_test_split(X_list[0], y_list[0], test_size=0.1, random_state=0)
X_df_1=X_list[1]
y_df_1=y_list[1]
def x_select(x_i):
    from sklearn.ensemble import GradientBoostingRegressor

    gbdt = GradientBoostingRegressor(

        loss='ls'
        , learning_rate=0.1
        , n_estimators=200
        , subsample=1
        , min_samples_split=2
        , min_samples_leaf=50
        , max_depth=x_i
        , init=None
        , random_state=None
        , max_features=None
        , alpha=0.9
        , verbose=0
        , max_leaf_nodes=None
        , warm_start=False
    )
    model = gbdt.fit(X_train, y_train)
    # print(model)
    print('max_depth :', i)

    train_score = model.score(X_train, y_train)
    print('train_score', train_score)
    test_score = model.score(X_test, y_test)
    print('test_score :', test_score)
    y_pred = abs(gbdt.predict(X_test))

    res = y_pred - y_test
    accuracy(y_pred, y_test, res,x_i)
    ##将模型存入pickle或者joblib,下次就不用重复训练了
    # import pickle
    # from sklearn.externals import joblib
    #
    # f = open('gbdt_20170527.sav', 'wb')
    # pickle.dump(gbdt,f)
    # f.close()
    # f2 = open('gbdt_20170527.sav', 'rb')
    #
    # GBDT = pickle.load(f2)
    # f2.close()


    y_pred_df_1 = abs(gbdt.predict(X_df_1))

    res = y_pred_df_1 - y_df_1
    accuracy(y_pred_df_1, y_df_1, res, x_i)

    # 计算
    from sklearn import metrics
    # confidence_0=np.multiply(y_test, 0.2)
    # confidence_1=np.multiply(y_test, -0.2)
    # print("RMSE by hand:", '%.2f' % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  # 保留两位小数

    print('---------------------------------')
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(range(len(y_pred)), y_pred, '+', label="predict")
    # plt.plot(range(len(y_pred)), y_test, 'x', color='r', label="real value")
    # # plt.plot(range(len(y_pred)),res,'b',label="residual error")
    # plt.legend(loc="upper right")  # 显示图中的标签
    # plt.xlabel("the time of predict")
    # plt.ylabel('value of ds')
    # plt.show()

x_dict={}
for i in range(7,8):
    x_select(i)
print(x_dict.items())
print()


# a=abs(res)<y_test*3
# y_pred.astype(np.int32)
# print(pd.value_counts(y_pred))
# print(pd.value_counts(a))
# print(np.count_nonzero(a))

# print(res<y_test*0.2)
# print(type(res<10))





