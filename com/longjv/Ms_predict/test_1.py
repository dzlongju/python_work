import numpy as np
import pandas as pd
df=pd.read_csv('C:\\Users\\duozhun\\Desktop\\日消数据\\测试数据\\re_item_id_ds_1.csv')
from sklearn.model_selection  import train_test_split
df_esle,df= train_test_split(df,test_size=0.9)
# print(df.count())
# lrr=lrr[lrr.dm_1 > 0]
# df=df.dropna()


featue_base=df[['dm_1','dm_2','dm_3','dm_4','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']]


df['dm_mean']=featue_base.mean(axis=1)

# X_1=df[['dm_mean','dm_1','dm_2','dm_3','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']]
X=df[['dm_1','dm_2','dm_3','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']]
# X=df[['dm_1','dm_2','dm_3','dm_4','dm_5','dm_6','dm_7','dm_8','dm_9','dm_10','dm_11','dm_12','dm_13','dm_14','dm_15','dm_16','dm_17','dm_18','dm_19','dm_20','dm_21','dm_22','dm_23','dm_24','dm_25','dm_26','dm_27','dm_28','dm_29','dm_30']]
y=df['ds']

# featue_base_diff = featue_base.diff(periods=1, axis=1).drop('dm_1', 1)
# featue_base_diff=pd.DataFrame(np.where(featue_base_diff > 0,featue_base_diff,0))
#
# print(featue_base_diff)
# print(type(featue_base))
# print(type(featue_base_diff))


# #确定每条记录的活动状态
# df['active_status'] = df.where(df['dt'].weekday()>4,1,0)
import time,datetime
from datetime import datetime
#输出日期的星期X格式,判断是否为周末
df['weekday'] = df['dt'].apply(lambda x:datetime.strptime(datetime.strptime(x, "%Y/%m/%d").strftime("%Y%m%d"),'%Y%m%d').weekday())
df['month']= df['dt'].apply(lambda x:datetime.strptime(datetime.strptime(x, "%Y/%m/%d").strftime("%Y%m%d"),'%Y%m%d').month)
#one-hot处理

#利用get_dummies里面的prefix参数可以给哑变量矩阵加上定制标签名
month_dummies=pd.get_dummies(df['month'],prefix='month',prefix_sep='')
weekday_dummies = pd.get_dummies(df['weekday'],prefix='weekday')
monthname_list=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
weekdayname_list=['星期一','星期二','星期三','星期四','星期五','星期六','星期天']

# month_dummies.rename(columns=lambda x: x+'month', inplace=True)
# for i in month_dummies:
#     month_dummies.rename(columns={i:monthname_list[int(i-1)]}, inplace = True)
# for i in weekday_dummies:
#     weekday_dummies.rename(columns={i:weekdayname_list[int(i-1)]}, inplace = True)
print(month_dummies)



# print(weekday_dummies)

# from sklearn import svm
# from sklearn import datasets
# clf=svm.SVC()
# iris=datasets.load_iris()
# X,y=iris.data,iris.target
# clf.fit(X,y)


# df['weekday'] = (df['weekday'].where(df['weekday']< 4,1)).where(df['weekday']> 4,3)
# i= '2015/1/1'
# dt = time.strptime(datetime.strptime(i, "%Y/%m/%d").strftime("%Y%m%d"),'%Y%m%d')
# print(df['weekday'])
# print(df)
#
# timeArray=[]
# import time,datetime
# for i in df['dt']:
#     time1 = datetime.datetime.strptime(i, '%Y%m%d')
#     timeArray.append(time1)
# print(timeArray)
#
# for i in featue_base_diff:
#     featue_base_diff.rename(columns={i:i+'_diff'}, inplace = True)
# X=X.join(featue_base_diff)


# dt = datetime.datetime.now()
# print(dt.weekday())

# print(featue_base['dm_3']-featue_base['dm_2'])
# from numpy import log1p
# from sklearn.preprocessing import FunctionTransformer
#
# log_feature=FunctionTransformer(log1p).fit_transform(featue_base)
# df_log = pd.DataFrame(log_feature)
# X_2= X_1.join(df_log)



#K折交叉验证 -- 线性回归
# def get_score(X,y):
#     from sklearn.linear_model import LinearRegression
#     linreg = LinearRegression()
#     from sklearn.model_selection import cross_val_score
#     score = cross_val_score(linreg, X, y, cv=5)
#     print("score=", score)
# get_score(X,y)
# get_score(X_1,y)
# get_score(X_2,y)


#K折交叉验证 -- GBDT
# def get_score(X,y):
#     from sklearn.ensemble import GradientBoostingRegressor
#
#     gbdt = GradientBoostingRegressor(
#
#         loss='ls'
#         , learning_rate=0.1
#         , n_estimators=200
#         , subsample=1
#         , min_samples_split=2
#         , min_samples_leaf=30
#         , max_depth=7
#         , init=None
#         , random_state=None
#         , max_features=None
#         , alpha=0.9
#         , verbose=0
#         , max_leaf_nodes=None
#         , warm_start=False
#     )
#     from sklearn.model_selection import cross_val_score
#     score = cross_val_score(gbdt, X, y, cv=5)
#     print("score=", score)
# get_score(X,y)
# get_score(X_1,y)
# get_score(X_2,y)

#
# from splinter.browser import Browser
# x = Browser(driver_name="chrome")
# url = 'https://kyfw.12306.cn/otn/leftTicket/ini'
# x = Browser(driver_name="chrome")
# x.visit(url)
# #填写登陆账户、密码
# x.find_by_text(u"登录").click()
# x.fill("loginUserDTO.user_name","账号(少游sb)")
# x.fill("userDTO.password","密码")
# #填写出发点目的地
# x.cookies.add({"_jc_save_fromStation":"杭州东"})
# x.cookies.add({"_jc_save_fromDate":"2016-01-20"})
# x.cookies.add({u'_jc_save_toStation':'温州南'})
# #加载查询
# x.reload()
# x.find_by_text(u"查询").click()
# #预定
# x.find_by_text(u"预订")[1].click()
# #选择乘客
# x.find_by_text(u"高飞")[1].click()


# #-*- coding:utf-8 -*-
# #京东抢手机脚本
# from splinter.browser import Browser
# import time
#
# #登录页
# def login(b):  #登录京东
#     b.click_link_by_text("你好，请登录")
#     time.sleep(3)
#     b.fill("loginname","account*****")  #填写账户密码
#     b.fill("nloginpwd","passport*****")
#     b.find_by_id("loginsubmit").click()
#     time.sleep(3)
#     return b
#
# #订单页
# def loop(b):  #循环点击
#     try:
#         if b.title=="订单结算页 -京东商城":
#             b.find_by_text("保存收货人信息").click()
#             b.find_by_text("保存支付及配送方式").click()
#             b.find_by_id("order-submit").click()
#             return b
#         else:  #多次抢购操作后，有可能会被转到京东首页，所以要再打开手机主页
#             b.visit("http://item.jd.com/2707976.html")
#             b.find_by_id("choose-btn-qiang").click()
#             time.sleep(10)
#             loop(b)  #递归操作
#     except Exception as e: #异常情况处理，以免中断程序
#         b.reload()  #重新刷新当前页面，此页面为订单提交页
#         time.sleep(2)
#         loop(b)  #重新调用自己
#
#
# b=Browser(driver_name="chrome") #打开浏览器
# b.visit("http://item.jd.com/2707976.html")
# login(b)
# b.find_by_id("choose-btn-qiang").click() #找到抢购按钮，点击
# time.sleep(10)  #等待10sec
# while True:
#     loop(b)
#     if b.is_element_present_by_id("tryBtn"): #订单提交后显示“再次抢购”的话
#         b.find_by_id("tryBtn").click()  #点击再次抢购，进入读秒5，跳转订单页
#         time.sleep(6.5)
#     elif b.title=="订单结算页 -京东商城": #如果还在订单结算页
#         b.find_by_id("order-submit").click()
#     else:
#         print('恭喜你，抢购成功')
#         break