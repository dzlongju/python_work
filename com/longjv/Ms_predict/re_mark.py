import os
import re
import glob
import pandas
import numpy as np
import datetime

# 将源数据_店铺提取文件夹里所有的文本文件数据输出成一个总的csv或者文本
# filenames=glob.glob('C:\\Users\\duozhun\\Desktop\\日消数据\\源数据_店铺提取\\*.txt')
# fp = open('C:\\Users\\duozhun\\Desktop\\副本数据.txt','w')
# for filename in filenames:
#     lines = open(filename,'r')  #打开文件，读入每一行
#     for f in lines:
#          message = f.replace('\t',',')
#          message = f.replace('dt, seller_id, auction_id, thedate, paycnt, paynum', '')
#          fp.write(message)
# fp.close()



class day_data(object):
    def __init__(self,auction_id,list_data):
        self.auction_id = auction_id
        self.list_data = list_data

d_begin = datetime.datetime.strptime('20140101', '%Y%m%d')
d_end = datetime.datetime.strptime('20170401', '%Y%m%d')
day_all =int((d_end -d_begin).days)

##判断并转化数组中的-1的日消=0
def panduan(a):
    a = int(a)
    if a != -1:
        return a
    else:
        return 0

##计算每一天的月销量
def calculate_sum(l):
    new_l = []
    new_l.append(panduan(l.list_data[0]))
    for i in range(len(l.list_data)):
        if i < 30:
            new_l.append(panduan(new_l[len(new_l) - 1]) + panduan(l.list_data[i]))
        else:
            new_l.append(panduan(new_l[len(new_l) - 1]) + panduan(l.list_data[i]) - panduan(l.list_data[i - 30]))
            # return new_l
    print(l.auction_id,':',new_l)
    #     m1, m2 = l.auction_id, str(new_l)
    # fp2.write(m1)
    # fp2.write(':')
    # fp2.write(m2+'\n')

lines = open('C:\\Users\\duozhun\\Desktop\\data_get.txt', 'r')
data_all,data_in={},{}
for f in lines:
    s = re.split(',', f.strip())
    data_in[s[3]]= s[4]
    data_all[s[2]] = s[3]

##将数据转化成一个item_id对应所有日期数据的数组
def transform(item_id):
    n,day_data_all = [],[]
    m = [None] * day_all
    for i in range(len(m)):
        n.append(-1)

    for f in data_all:
        if f==item_id:
            d_now = datetime.datetime.strptime(data_all[f], '%Y%m%d')
            j =int((d_now-d_begin).days)
            n[j]=int(data_in[data_all[f]])
    print(n)
    a = day_data(item_id,n)
    m1, m2 =a.auction_id,str(a.list_data)
    # fp1.write(m1)
    # fp1.write(':')
    # fp1.write(m2+'\n')
    # print(a.auction_id,a.list_data)
    calculate_sum(a)

lines = open('C:\\Users\\duozhun\\Desktop\\日消数据\\item_id提取.txt','r')
for f in lines:
    dis_itemid = re.split(',', f.strip())

# for f in lines:
#     dis_itemid.append(f)


fp1 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\item_id_ds_10000_3.txt','w')
fp2 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\item_id_ms_10000_3.txt','w')
transform('13505587887')
# for list_i in range(len(dis_itemid)):
#     transform(dis_itemid[list_i])



