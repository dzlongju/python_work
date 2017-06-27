import pandas as pd
import  os
#预测数据特征处理
def feature(path,f):
    # df=pd.read_csv(path,encoding='utf-8')
    df=pd.read_csv(path + '\\' + f,encoding='utf-8')
    df=df.dropna()
    feature_base = df[['index']]
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
    return X, df

#预测并输出结果.csv
def predict(X,df):
    f2 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\日销售指数预测\\zhishufanyi_gbdt_20170612.sav', 'rb')
    import pickle
    GBDT = pickle.load(f2)
    y_pred = GBDT.predict(X)
    # dd = pd.DataFrame(y.tolist())
    df['pred'] = pd.DataFrame(y_pred).astype('int')
    # dd['shopname']=dd[0]
    # dd[['shopname', 'y_pred']].to_csv('C:\\Users\\duozhun\\Desktop\\生意参谋指数储存\\aaa.csv')
    # print(df[['from','shopname','value','index','rate','pred']])
    f2.close()

# #特征处理
# path2='C:\\Users\\duozhun\\Desktop\\生意参谋指数储存\\整理\\18_20\\tideword.csv'
# X,y=feature(path2)
# predict(X,y)


path = 'C:\\Users\\duozhun\\Desktop\\生意参谋指数储存\\整理\\18_20\\访客来源结构\\'
files = os.listdir(path)
for f in files:
    X,df=feature(path,f)
    predict(X,df)
    df[['from', 'shopname', 'value', 'index', 'rate', 'pred']].to_csv(path +'\\'+f)
