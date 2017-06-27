import numpy as np
import pandas as pd
df=pd.read_csv('C:\\Users\\duozhun\\Desktop\\钻展合理投放测试数据\\女装店铺数据.csv')

# print(df['consume'].mean())
roi_mean = sum(df['15roi']*df['consume'])/sum(df['consume'])
consume_mean = df['consume'].mean()

df['welldone'] = 0
df['welldone']=df['welldone'].where(df['consume']< consume_mean ,3,2)
df['welldone']=df['welldone'].where(df['15roi']< roi_mean ,1,0)
# df['welldone']=df['welldone'].where(df['consume']> consume_mean ,0)
# print((df[df['welldone']>0])['welldone'])
print(df[['shop_name','consume','15roi','welldone']])


# def accuracy(y_pred, y_test):
#   num_state_1,num_state_0=0,0
#   y_pred=y_pred.tolist()
#   y_test =y_test.tolist()
#   for i in range(len(y_pred)):
#    if y_test[i] ==y_pred[i]:
#     num_state_1=num_state_1+1
#    else:num_state_0=num_state_0+1
#   accu =num_state_1 / len(y_pred)
#   print("accuracy =",'%.4f'%accu)
#
# X=df[['add_cart', 'add_fav', 'clicks','views','cost_fav', 'cost_cart', 'cpc', 'cpm', 'clicks_rate', 'fav_rate','cart_rate']]
# # X=df[['dm_1','dm_2','dm_3','dm_4','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']]
# y=df['welldone']
# # df['welldone'] = df['15roi'].apply(lambda x:datetime.strptime(datetime.strptime(x, "%Y/%m/%d").strftime("%Y%m%d"),'%Y%m%d').weekday())
#
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection  import train_test_split
# X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=0)
# alg = GradientBoostingClassifier()
# results = []
# # 最小叶子结点的参数取值
# max_depth_options=list(range(1,7))
# sample_leaf_options =list(range(1,10,1))
# # 决策树个数参数取值
# n_estimators_options = list(range(1,10,1))
# groud_truth = y_test
#
# for leaf_size in sample_leaf_options:
#     for n_estimators_size in n_estimators_options:
#         alg = GradientBoostingClassifier(
#              learning_rate=0.1
#             , n_estimators=n_estimators_size
#             , min_samples_split=2
#             , min_samples_leaf=1
#             , max_depth=6
#             , random_state=None
#         )
#         alg.fit(X_train, y_train)
#         predict = alg.predict(X_test)
#         # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
#         results.append((leaf_size, n_estimators_size, (groud_truth == predict).mean()))
#         # 真实结果和预测结果进行比较，计算准确率
#         print((groud_truth == predict).mean())
#
# # 打印精度最大的那一个三元组
# print(max(results, key=lambda x: x[2]))

