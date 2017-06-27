import numpy as np
import pandas as pd

#准确判定条件
def accuracy(res_rate):
  num_state=0
  res_rate=res_rate.tolist()
  for i in range(len(res_rate)):
   if res_rate[i] < 0.01:
    num_state=num_state+1

  accu =num_state / len(res_rate)
  print("accuracy =",'%.2f%%'%(accu*100))

#特征处理
def feature(path):
    df=pd.read_csv(path)
    df=df.dropna()
    feature_base = df[['sale_index']]
    bins = []
    for i in range(30):
        x = i * 0.1 + 0.1
        bins.append(float(x))
    # print(featue_base**0.1)
    x = 0.75
    for j in range(16):
        if x < 2.5:
            x = x + 0.1
            bins.append(x)
    for i in range(len(bins)):
        df['sale_index_' + str(i)] = feature_base ** bins[i]

    X = df[['sale_index_0', 'sale_index_1', 'sale_index_2',
            'sale_index_3', 'sale_index_4', 'sale_index_5', 'sale_index_6',
            'sale_index_7', 'sale_index_8', 'sale_index_9', 'sale_index_10',
            'sale_index_11', 'sale_index_12', 'sale_index_13', 'sale_index_14',
            'sale_index_15', 'sale_index_16', 'sale_index_17', 'sale_index_18',
            'sale_index_19', 'sale_index_20', 'sale_index_21', 'sale_index_22',
            'sale_index_23', 'sale_index_24', 'sale_index_25', 'sale_index_26',
            'sale_index_27', 'sale_index_28', 'sale_index_29', 'sale_index_30',
            'sale_index_31', 'sale_index_32', 'sale_index_33', 'sale_index_34',
            'sale_index_35', 'sale_index_36', 'sale_index_37', 'sale_index_38',
            'sale_index_39', 'sale_index_40', 'sale_index_41', 'sale_index_42',
            'sale_index_43', 'sale_index_44', 'sale_index_45']]
    y = df['sale']
    return X, y

#预测数据特征处理
# def feature2(path):
#     df=pd.read_csv(path,encoding='utf-8')
#     df=df.dropna()
#     feature_base = df[['sale_index']]
#     bins = []
#     for i in range(30):
#         x = i * 0.1 + 0.1
#         bins.append(float(x))
#     # print(featue_base**0.1)
#     x = 0.75
#     for j in range(16):
#         if x < 2.5:
#             x = x + 0.1
#             bins.append(x)
#     for i in range(len(bins)):
#         df['sale_index_' + str(i)] = feature_base ** bins[i]
#
#     X = df[['sale_index_0', 'sale_index_1', 'sale_index_2',
#             'sale_index_3', 'sale_index_4', 'sale_index_5', 'sale_index_6',
#             'sale_index_7', 'sale_index_8', 'sale_index_9', 'sale_index_10',
#             'sale_index_11', 'sale_index_12', 'sale_index_13', 'sale_index_14',
#             'sale_index_15', 'sale_index_16', 'sale_index_17', 'sale_index_18',
#             'sale_index_19', 'sale_index_20', 'sale_index_21', 'sale_index_22',
#             'sale_index_23', 'sale_index_24', 'sale_index_25', 'sale_index_26',
#             'sale_index_27', 'sale_index_28', 'sale_index_29', 'sale_index_30',
#             'sale_index_31', 'sale_index_32', 'sale_index_33', 'sale_index_34',
#             'sale_index_35', 'sale_index_36', 'sale_index_37', 'sale_index_38',
#             'sale_index_39', 'sale_index_40', 'sale_index_41', 'sale_index_42',
#             'sale_index_43', 'sale_index_44', 'sale_index_45']]
#     y = df['shopname']
#     return X, y

#预测并输出结果.csv
def predict(X,y):
    f2 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\日销售指数预测\\zhishufanyi_gbdt_20170612.sav', 'rb')
    import pickle
    GBDT = pickle.load(f2)
    y_pred = GBDT.predict(X)
    dd = pd.DataFrame(y.tolist())
    dd['y_pred'] = pd.DataFrame(y_pred).astype('int')
    # dd['res'] = dd['y_pred'] - dd[0]
    # dd['res/real'] = abs(dd['res'] / dd[0])
    dd['sale'] = dd[0]
    # dd['shopname']=dd[0]
    # dd.columns=['真实值','预测值','误差值','误差率']
    from sklearn import metrics
    # print("RMSE by hand:", '%.2f' % np.sqrt(metrics.mean_squared_error(y, y_pred)))
    # print(dd)
    # print(dd[['shopname','y_pred']])
    dd[['sale', 'y_pred']].to_csv('C:\\Users\\duozhun\\Desktop\\生意参谋指数储存\\aaa.csv')
    # dd[['shopname','y_pred']].to_csv('C:\\Users\\duozhun\\Desktop\\日消数据\\日销售指数预测\\商务需求_行业top500\\aaa.csv')
    # print('假定误差率绝对值小于1%,认定预测准确,则上述模型的预测准确率达', ':')
    # accuracy(dd['res/real'])
    # dd.to_csv('C:\\Users\\duozhun\\Desktop\\日消数据\\日销售指数预测\\预测效果对比\\女装行业交易指数测试.csv')
    f2.close()


##训练模型
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection  import train_test_split
# #交易指数训练集
# path1='C:\\Users\\duozhun\\Desktop\\日消数据\\日销售指数预测\\train.csv'
# # #访客指数训练集
# # path1='C:\\Users\\duozhun\\Desktop\\日消数据\\日销售指数预测\\train_view.csv'
# X,y=feature(path1)
# X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=0)
# alg = GradientBoostingRegressor()
# results = []
# # 最小叶子结点的参数取值
# max_depth_options=list(range(1,2,1))
# sample_leaf_options =list(range(1,2,1))
# # 决策树个数参数取值
# n_estimators_options = list(range(100,110,10))
# groud_truth = y_test

# for leaf_size in sample_leaf_options:
#     for n_estimators_size in n_estimators_options:
#         gbdt = GradientBoostingRegressor(
#              learning_rate=0.1
#             , n_estimators=n_estimators_size
#             , min_samples_split=2
#             , min_samples_leaf=1
#             , max_depth=7
#             , random_state=None
#         )
#         gbdt.fit(X_train, y_train)
#         y_pred = gbdt.predict(X_test)
#         test_score=gbdt.score(X_test, y_test)
#         results.append((leaf_size, n_estimators_size,test_score ))
#         # 真实结果和预测结果进行比较，计算准确率
#         # print(test_score)
#         # dd=pd.DataFrame(y_test.tolist())
#         # dd['y_pred']=pd.DataFrame(y_pred).astype('int')
#         # dd['res'] = dd['y_pred'] - dd[0]
#         # dd['res/real_sale'] = abs(dd['res'] / dd[0])
#         # print(dd)
#         #将模型存入pickle或者joblib,下次就不用重复训练了
#         import pickle
#         from sklearn.externals import joblib
#
#         f = open('C:\\Users\\duozhun\\Desktop\\日消数据\\日销售指数预测\\zhishufanyi_gbdt_20170612.sav', 'wb')
#         pickle.dump(gbdt,f)
#         f.close()
# #
#
#
#
# # 打印精度最大的那一个三元组
# best_in =max(results, key=lambda x: x[2])
# print(best_in)
# print('在此参数组合下,模型预测准确率最高达:','%.2f%%'%(best_in[2]*100))


#特征处理
path2='C:\\Users\\duozhun\\Desktop\\日消数据\\日销售指数预测\\女装行业交易指数测试_15.csv'
# path2='C:\\Users\\duozhun\\Desktop\\日消数据\\日销售指数预测\\商务需求_行业top500\\top500男装店铺622销售指数.csv'
X,y=feature(path2)
# X,y=feature2(path2)


predict(X,y)