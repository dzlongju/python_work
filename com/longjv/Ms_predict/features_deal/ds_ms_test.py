

import re
import datetime

class line_data(object):
    def __init__(self,auction_id,dt,ds,list_data):
        self.auction_id = auction_id
        self.dt = dt
        self.ds = ds
        self.list_data = list_data

fp1 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\测试数据\\item_id_ds_10000.txt','r')
fp2 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\测试数据\\item_id_ms_10000.txt','r')


d_begin = datetime.datetime.strptime('20140101', '%Y%m%d')

dict_ds,dict_dm={},{}

for f in fp1:
    s = re.split(':', f.strip())
    a1 = s[1].replace('[', '')
    a2 = a1.replace(']', '')
    a2 = a2.replace(' ', '')
    a3 = re.split(',', a2.strip())
    dict_ds[s[0]] = a3


for f in fp2:
    s = re.split(':', f.strip())
    a1 = s[1].replace('[', '')
    a2 = a1.replace(']', '')
    a2 = a2.replace(' ', '')
    a3 = re.split(',', a2.strip())
    dict_dm[s[0]] = a3


def match_dm(item_id,dt_w):
    dm_list=[]
    a=dict_dm[item_id]
    if dt_w < 1156:
        for i in range(dt_w+1, dt_w + 31):
            dm_list.append(a[i])
            fp3.write(a[i])
            fp3.write(',')




d_begin = datetime.datetime.strptime('20140101', '%Y%m%d')


try:
        fp3 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\测试数据\\re_item_id_ds.csv','w')
        for f in dict_ds:
            ds_item_id=f
            ds_list = dict_ds[f]
            # print(ds_item_id,':',a3)
            # print(a3[0],',',a3[1])
            for w in range(len(ds_list)):
                if ds_list[w] != '-1':
                    delta = datetime.timedelta(days=w)
                    dt = d_begin+delta
                    dt=dt.strftime('%Y-%m-%d')
                    ds = ds_list[w]
                    fp3.write(ds_item_id)
                    fp3.write(',')
                    fp3.write(dt)
                    fp3.write(',')
                    fp3.write(ds)
                    fp3.write(',')
                    #运行match_dm方法
                    match_dm(ds_item_id, w)
                    fp3.write('\n')

finally:
    if fp3:
        fp3.close()