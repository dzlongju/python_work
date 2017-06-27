

#对特征进行归一化
from sklearn.preprocessing import Normalizer
X=Normalizer().fit_transform(X.astype(float))


# #对特征进行对数转化
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
X=FunctionTransformer(log1p).fit_transform(X)


#对特征进行标准化
from sklearn.preprocessing import StandardScaler
StandardScaler().fit_transform(X)

#对特征进行区间缩放法
from sklearn.preprocessing import MinMaxScaler
MinMaxScaler().fit_transform(X)


#对特征进行定量二值化
from sklearn.preprocessing import Binarizer
Binarizer(threshold=3).fit_transform(iris.data)


#K折交叉验证
from sklearn.model_selection  import cross_val_score
score = cross_val_score(model, X,y, cv=5)
print("score=",score)

#哑变量
import pandas as pd
import numpy
get.dummies( data[‘SHabit’] , prefix=’SHabit’)
data_noDup_rep_dum =pd.merge(data_noDup_rep, pd.get_dummies(data_noDup_rep[‘SHabit’],prefix=’SHabit’ ),right_index=True, left_index=True)

#对特征进行差分,axis(1为列差分,0为行差分)
featue_base_diff = featue_base.diff(periods=1, axis=1)

#给dataframe添加分层索引
import numpy as np
frame = pd.DataFrame(np.arange(12).reshape((4,3)),index=[['a','b','c','d'],[1,2,1,2]],
                     columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']])


#One-Hot 编码


