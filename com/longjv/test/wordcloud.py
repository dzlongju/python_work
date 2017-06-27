import matplotlib.pyplot as plt
# from wordcloud import WordCloud
import wordcloud
import jieba.analyse
jieba.load_userdict('C:\\Users\\duozhun\\AppData\\Local\\Programs\\Python\\Python35-32\\Lib\\site-packages\\jieba\\posseg\\userdict')#加载外部 用户词典

infile = open('C:\\Users\\duozhun\\Desktop\\银行大数据应用场景.txt','r', encoding='utf-8').read()

# infile=''
# word_list = jieba.analyse.extract_tags(infile, topK=100, withWeight=False)
word_list = jieba.cut(infile, cut_all=False)
wl_list = "/".join(word_list)
print(wl_list)
#确定
# my_wordcloud=wordcloud.wordcloud.WordCloud(font_path='./fonts/simhei.ttf').generate(wl_list)
# plt.imshow(my_wordcloud)
# plt.axis('off')
# plt.show()