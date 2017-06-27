import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
df=pd.read_csv('C:\\Users\\duozhun\\Desktop\\钻展合理投放测试数据\\女装店铺数据.csv')

# print(df.describe())
# roi_mean = sum(df['15roi']*df['consume'])/sum(df['consume'])
# consume_mean = df['consume'].mean()
#
# df['welldone'] = 0
# df['welldone']=df['welldone'].where(df['15roi']< roi_mean ,1)
# df['welldone']=df['welldone'].where(df['consume']> consume_mean ,0)
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
print('定义投放合格的店铺标签为1,不合格为0,检查分布如下:')
print('-------------------------------------------')
print(pd.value_counts(df['welldone']))

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

X=df[['add_cart', 'add_fav', 'clicks', 'views','cost_fav', 'cost_cart', 'cpc', 'cpm', 'clicks_rate', 'fav_rate','cart_rate']]
y=df['welldone']
# df['welldone'] = df['15roi'].apply(lambda x:datetime.strptime(datetime.strptime(x, "%Y/%m/%d").strftime("%Y%m%d"),'%Y%m%d').weekday())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=0)
# alg = RandomForestClassifier()
# model = alg.fit(X_train, y_train)
# test_score = model.score(X_test, y_test)
# print('test_score :', test_score)
# y_pred=alg.predict(X_test)
# accuracy(y_test,y_pred)
# # print(y_test,y_pred)


results = []
# 最小叶子结点的参数取值
sample_leaf_options = list(range(2,3,1))
# 决策树个数参数取值
n_estimators_options = list(range(70, 71, 1))
groud_truth = y_test

for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)
        alg.fit(X_train, y_train)
        predict = alg.predict(X_test)
        # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
        results.append((leaf_size, n_estimators_size,(groud_truth == predict).mean()))
        # 真实结果和预测结果进行比较，计算准确率
        print((groud_truth == predict).mean())


# 打印精度最大的那一个三元组
best_in =max(results, key=lambda x: x[2])
print(best_in)
print('在此参数组合下,模型预测准确率最高达:','%.2f%%'%(best_in[2]*100))

print('Test Accuracy: %.2f' %alg.score(X_test, y_test))
pred_probas = alg.predict_proba(X_test)[:, 1]
    # plot ROC curve
    # AUC = 0.92
    # KS = 0.7
fpr, tpr, _ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.xlabel('')
plt.show()