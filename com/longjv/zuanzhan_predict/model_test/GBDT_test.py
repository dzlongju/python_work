import numpy as np
import pandas as pd
df=pd.read_csv('C:\\Users\\duozhun\\Desktop\\钻展合理投放测试数据\\女装店铺数据.csv')

# print(df['consume'].mean())
# roi_mean = sum(df['15roi']*df['consume'])/sum(df['consume'])
# consume_mean = df['consume'].mean()
#按bins做消耗分层
bins=[0,5000,10000,25000,50000,100000]
roi_mean_list=[]
for i in range(len(bins)):
    if i != 5:
     roi_mean = sum((df['15roi'])[(df['consume']<bins[i+1]) & (df['consume']>bins[i])] * (df['consume'])[(df['consume']<bins[i+1]) & (df['consume']>bins[i])]) /\
               sum((df['consume'])[(df['consume']<bins[i+1]) & (df['consume']>bins[i])])
     roi_mean_list.append(roi_mean)

# for i in range(len(bins)):
#     if i != 5:
#         df['welldone'] = np.where(
#             (df['consume'] < bins[i + 1]) & (df['consume'] > bins[i]) & (df['15roi'] > roi_mean_list[i]), 1, 0)
#         print(df['welldone'])
df['welldone'] = np.where(
    (df['consume'] < bins[1]) & (df['consume'] > bins[0]) & (df['15roi'] > roi_mean_list[0]), 1,
    np.where((df['consume'] < bins[2]) & (df['consume'] > bins[1]) & (df['15roi'] > roi_mean_list[1]),1,
             np.where((df['consume'] < bins[3]) & (df['consume'] > bins[2]) & (df['15roi'] > roi_mean_list[2]),1,
                      np.where((df['consume'] < bins[4]) & (df['consume'] > bins[3]) & (df['15roi'] > roi_mean_list[3]),1,
                               np.where((df['consume'] < bins[5]) & (df['consume'] > bins[4]) & (df['15roi'] > roi_mean_list[4]), 1,0
                                        )))))
# print(df['welldone'])
print('定义投放合格的店铺标签为1,不合格为0,检查分布如下:')
print('-------------------------------------------')
print(pd.value_counts(df['welldone']))


# df_cut= pd.cut(np.array(df['consume']),bins,precision=2)
#
# print(pd.value_counts(df_cut))

# df['welldone_consume'] = 0
# df['welldone']=np.where(df['15roi']> roi_mean,1,0)
# print(df[['shop_name','consume','15roi','welldone']])
# print((df[['shop_name','consume','15roi','welldone']])[df['consume']<50000])


# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot((df[df['consume']<100000])['consume'], (df[df['consume']<100000])['15roi'], '+r')
# plt.legend(loc="upper right")
# plt.xlabel("consume")
# plt.ylabel('15day_roi')
# plt.show()
#
def accuracy(y_pred, y_test):
  num_state_1,num_state_0=0,0
  y_pred=y_pred.tolist()
  y_test =y_test.tolist()
  for i in range(len(y_pred)):
   if y_test[i] ==y_pred[i]:
    num_state_1=num_state_1+1
   else:num_state_0=num_state_0+1
  accu =num_state_1 / len(y_pred)
  print("accuracy =",'%.4f'%accu)

X=df[['add_cart', 'add_fav', 'clicks','views','cost_fav', 'cost_cart', 'cpc', 'cpm', 'clicks_rate', 'fav_rate','cart_rate']]
y=df['welldone']
# df['welldone'] = df['15roi'].apply(lambda x:datetime.strptime(datetime.strptime(x, "%Y/%m/%d").strftime("%Y%m%d"),'%Y%m%d').weekday())

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection  import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=0)
alg = GradientBoostingClassifier()
results = []
# 最小叶子结点的参数取值
max_depth_options=list(range(1,7))
sample_leaf_options =list(range(1,3,1))
# 决策树个数参数取值
n_estimators_options = list(range(1,10,1))
groud_truth = y_test

for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        alg = GradientBoostingClassifier(
             learning_rate=0.1
            , n_estimators=n_estimators_size
            , min_samples_split=2
            , min_samples_leaf=1
            , max_depth=6
            , random_state=None
        )
        alg.fit(X_train, y_train)
        predict = alg.predict(X_test)
        # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
        results.append((leaf_size, n_estimators_size, (groud_truth == predict).mean()))
        # 真实结果和预测结果进行比较，计算准确率
        print((groud_truth == predict).mean())

# 打印精度最大的那一个三元组
best_in =max(results, key=lambda x: x[2])
print(best_in)
print('在此参数组合下,模型预测准确率最高达:','%.2f%%'%(best_in[2]*100))
#
