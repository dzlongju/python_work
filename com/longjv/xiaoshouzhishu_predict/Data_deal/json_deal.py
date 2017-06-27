import json
import  os
import re
# def split(filename):

# fp1 = open('C:\\Users\\duozhun\\Desktop\\生意参谋指数储存\\19_17点\\SleepyBunny瞌睡兔 - 2017-06-19 17-04-54 - 概览和竞店排行.json','r',encoding='utf-8')
# data = json.load(fp1)

# path='C:\\Users\\duozhun\\Desktop\\生意参谋指数储存\\19_12点'
# files = os.listdir(path)
# for f in files:
#     s = re.split(' - ', f.strip())
#     # print(s)
#     fp = open(path + '\\' + f, 'r', encoding='utf-8')
#     data = json.load(fp)
#
#     s[2]=s[2].replace('.json','')
#     filename = s[0] + '_' + s[2]
#     fp1 = open('C:\\Users\\duozhun\\Desktop\\生意参谋指数储存\\整理\\19_17\\'+filename+'.csv', 'w')
#
#     if s[2]=='竞店来源构成':
#         title_list = ['from', 'shopname', 'value', 'index', 'rate']
#         b = title_list
#         b = str(b).replace('[', '')
#         b = b.replace(']', '')
#         b = b.replace("'", '')
#         b = b.replace(" ", '')
#         fp1.write(b)
#         fp1.write('\n')
#         for i in range(len(data)):
#             for j in range(len(data[i]['list']['mobile'])):
#                 # print(data[i]['list']['pc'][j]['competitor'])
#                 b = []
#                 b.append('moblie')
#                 b.append(data[i]['title'])
#                 b.append(data[i]['list']['mobile'][j]['competitor']['sourceGroupName'])
#                 b.append(data[i]['list']['mobile'][j]['competitor']['index'])
#                 b.append(data[i]['list']['mobile'][j]['competitor']['rate'])
#                 b=str(b).replace('[','')
#                 b = b.replace(']', '')
#                 b = b.replace("'", '')
#                 b = b.replace(" ", '')
#                 fp1.write(b)
#                 fp1.write('\n')
#                 print(b)
#
#             for j in range(len(data[i]['list']['pc'])):
#                 # print(data[i]['list']['pc'][j]['competitor'])
#                 b = []
#                 b.append('pc')
#                 b.append(data[i]['title'])
#                 b.append(data[i]['list']['pc'][j]['competitor']['sourceGroupName'])
#                 b.append(data[i]['list']['pc'][j]['competitor']['index'])
#                 b.append(data[i]['list']['pc'][j]['competitor']['rate'])
#                 b = str(b).replace('[', '')
#                 b = b.replace(']', '')
#                 b = b.replace("'", '')
#                 b = b.replace(" ", '')
#                 fp1.write(b)
#                 fp1.write('\n')
#                 print(b)
#     else:
#         b = []
#         b.append('本店真实值')
#         b.append(data['summary']['uv']['my'])
#         b.append(data['summary']['payAmt']['my'])
#         b = str(b).replace('[', '')
#         b = b.replace(']', '')
#         b = b.replace("'", '')
#         b = b.replace(" ", '')
#         fp1.write(b)
#         fp1.write('\n')
#         print(b)
#         for i in range(len(data['payamtRank']['list'])):
#             c = []
#             shopname = data['payamtRank']['list'][i]['title']
#             c.append(shopname)
#             for j in range(len(data['uvRank']['list'])):
#                 if data['uvRank']['list'][j]['title'] == data['payamtRank']['list'][i]['title']:
#                     c.append(data['uvRank']['list'][j]['value'])
#                     break
#             c.append(data['payamtRank']['list'][i]['value'])
#             c = str(c).replace('[', '')
#             c = c.replace(']', '')
#             c=c.replace("'",'')
#             c = c.replace(" ", '')
#             fp1.write(c)
#             fp1.write('\n')
#             print(c)
#     fp1.close()
#     #copy fp1 to fp2 ,then try pd.read_csv(fp2) and predict the index to real value


# for i in range(len(data)):
#     a[i+1]=data[i]['title']
#
#     print(data[i]['title'])
# #
# print(a)

#解析访客来源
# for i  in range(len(data)):
#      for j in range(len(data[i]['list']['mobile'])):
#          # print(data[i]['list']['pc'][j]['competitor'])
#          b=[]
#          b.append(data[i]['title'])
#          b.append(data[i]['list']['mobile'][j]['competitor']['sourceGroupName'])
#          b.append(data[i]['list']['mobile'][j]['competitor']['index'])
#          b.append(data[i]['list']['mobile'][j]['competitor']['rate'])
#          print(b)
#
#      for j in range(len(data[i]['list']['pc'])):
#          # print(data[i]['list']['pc'][j]['competitor'])
#          b=[]
#          b.append(data[i]['title'])
#          b.append(data[i]['list']['pc'][j]['competitor']['sourceGroupName'])
#          b.append(data[i]['list']['pc'][j]['competitor']['index'])
#          b.append(data[i]['list']['pc'][j]['competitor']['rate'])
#          print(b)

#解析竞店排行
# b=[]
# b.append('本店真实值')
# b.append(data['summary']['uv']['my'])
# b.append(data['summary']['payAmt']['my'])
# print(b)
# for i in range(len(data['payamtRank']['list'])):
#     c=[]
#     shopname=data['payamtRank']['list'][i]['title']
#     c.append(shopname)
#     for j in range(len(data['uvRank']['list'])):
#         if data['uvRank']['list'][j]['title']==data['payamtRank']['list'][i]['title']:
#             c.append(data['uvRank']['list'][j]['value'])
#             break
#     c.append(data['payamtRank']['list'][i]['value'])
#     print(c)


path='C:\\Users\\duozhun\\Desktop\\生意参谋指数储存\\15点\\15点'
files = os.listdir(path)
for f in files:
    # print(s)
    fp = open(path + '\\' + f, 'r', encoding='utf-8')
    data = json.load(fp)
    f=f.replace('.json','')
    fp1 = open('C:\\Users\\duozhun\\Desktop\\生意参谋指数储存\\整理\\18_15\\访客来源结构\\'+f+'.csv', 'w',encoding='utf-8')
    title_list = ['from', 'shopname', 'value', 'index', 'rate']
    b = title_list
    b = str(b).replace('[', '')
    b = b.replace(']', '')
    b = b.replace("'", '')
    b = b.replace(" ", '')
    fp1.write(b)
    fp1.write('\n')
    for i in range(len(data)):
        for j in range(len(data[i]['list']['mobile'])):
            # print(data[i]['list']['pc'][j]['competitor'])
            b = []
            b.append('moblie')
            b.append(data[i]['title'])
            b.append(data[i]['list']['mobile'][j]['competitor']['sourceGroupName'])
            b.append(data[i]['list']['mobile'][j]['competitor']['index'])
            b.append(data[i]['list']['mobile'][j]['competitor']['rate'])
            b = str(b).replace('[', '')
            b = b.replace(']', '')
            b = b.replace("'", '')
            b = b.replace(" ", '')
            fp1.write(b)
            fp1.write('\n')
            print(b)

        for j in range(len(data[i]['list']['pc'])):
            # print(data[i]['list']['pc'][j]['competitor'])
            b = []
            b.append('pc')
            b.append(data[i]['title'])
            b.append(data[i]['list']['pc'][j]['competitor']['sourceGroupName'])
            b.append(data[i]['list']['pc'][j]['competitor']['index'])
            b.append(data[i]['list']['pc'][j]['competitor']['rate'])
            b = str(b).replace('[', '')
            b = b.replace(']', '')
            b = b.replace("'", '')
            b = b.replace(" ", '')
            fp1.write(b)
            fp1.write('\n')
            print(b)
