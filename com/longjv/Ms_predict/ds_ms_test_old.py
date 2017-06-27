import os
import re
import glob
import pandas
import numpy as np
import datetime


class line_data(object):
    def __init__(self, auction_id, dt, ds, list_data):
        self.auction_id = auction_id
        self.dt = dt
        self.ds = ds
        self.list_data = list_data


fp1 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\item_id_ds_10000_1.txt', 'r')

fp3 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\re_item_id_ds.txt', 'w')


def match_dm(item_id, dt_w):
    fp2 = open('C:\\Users\\duozhun\\Desktop\\日消数据\\item_id_ms_10000_1.csv', 'r')
    for f in fp2:
        s = re.split(':', f.strip())
        ds_item = s[0]
        a1 = s[1].replace('[', '')
        a2 = a1.replace(']', '')
        a2 = a2.replace(' ', '')
        a3 = re.split(',', a2.strip())
        dm_list = []
        if item_id == ds_item:
            for i in range(dt_w, dt_w + 29):
                dm_list.append(a3[i])
                fp3.write(a3[i])
                fp3.write(',')
            print(dm_list)


d_begin = datetime.datetime.strptime('20140101', '%Y%m%d')

try:
    for f in fp1:
        s = re.split(':', f.strip())
        ds_item_id = s[0]
        a1 = s[1].replace('[', '')
        a2 = a1.replace(']', '')
        a2 = a2.replace(' ', '')
        a3 = re.split(',', a2.strip())
        ds_list = a3
        # print(ds_item_id,':',a3)
        # print(a3[0],',',a3[1])
        for w in range(len(ds_list)):
            if ds_list[w] != '-1':
                delta = datetime.timedelta(days=w)
                dt = d_begin + delta
                dt = dt.strftime('%Y-%m-%d')
                ds = ds_list[w]
                fp3.write(ds_item_id)
                fp3.write(',')
                fp3.write(dt)
                fp3.write(',')
                fp3.write(ds)
                # 运行match_dm方法
                match_dm(ds_item_id, w)
                fp3.write('\n')


finally:
    if fp3:
        fp3.close()

