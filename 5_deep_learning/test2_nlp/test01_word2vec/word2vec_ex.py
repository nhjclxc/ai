#!/usr/bin/env python
# coding: utf-8

# 参考资料：<br>
# 白话word2vec：https://zhuanlan.zhihu.com/p/81032021 <br>
# 什么是词向量？https://blog.csdn.net/mawenqi0729/article/details/80698350 <br>
# 官方参数解读：https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.html#gensim.models.Word2Vec <br>
# 为什么PCA：https://zhuanlan.zhihu.com/p/37810506 <br>
# https://blog.csdn.net/HLBoy_happy/article/details/77146012 <br>
# 参数解读博客版：https://blog.csdn.net/xiaoQL520/article/details/102509477 <br>
# 负采样：https://zhuanlan.zhihu.com/p/144146838 <br>
# 更多资料：https://zhuanlan.zhihu.com/p/26306795 <br>
# https://mp.weixin.qq.com/s/j8JPMZSPoVT_hQswX5QVxA

# In[64]:


import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib
import torch


# ## 分词

# In[65]:


f = open("sanguo.txt", 'r',encoding='utf-8') #读入文本
lines = []
for line in f: #分别对每段分词
    temp = jieba.lcut(line)  #结巴分词 精确模式
    words = []
    for i in temp:
        #过滤掉所有的标点符号
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)
print(lines[0:5])#预览前5行分词结果


# ## 模型训练 

# In[158]:


# 调用Word2Vec训练
# 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(lines,vector_size = 20, window = 2 , min_count = 3, epochs=7, negative=10,sg=1)
print("孔明的词向量：\n",model.wv.get_vector('孔明'))
print("\n和孔明相关性最高的前20个词语：")
# 与孔明最相关的前20个词语
print(model.wv.most_similar('孔明', topn=20))

# 两个张量
x = torch.tensor([1, 2, 3]).float()
y = torch.tensor([2, 3, 6]).float()
print("余弦相似度:", torch.nn.functional.cosine_similarity(y, x, dim=0).item())

print("孔明 的词向量：\n",model.wv.get_vector('孔明'))
print("诸葛亮 的词向量：\n",model.wv.get_vector('诸葛亮'))
print("先主 的词向量：\n",model.wv.get_vector('先主'))
print("玄德 的词向量：\n",model.wv.get_vector('玄德'))
print("曹操 的词向量：\n",model.wv.get_vector('曹操'))
print('孔明和诸葛亮的余弦相似度: ', torch.nn.functional.cosine_similarity(
                    torch.tensor(model.wv.get_vector('孔明')),torch.tensor(model.wv.get_vector('诸葛亮')), dim=0).item())
print('孔明和先主的余弦相似度: ', torch.nn.functional.cosine_similarity(
                    torch.tensor(model.wv.get_vector('孔明')),torch.tensor(model.wv.get_vector('先主')), dim=0).item())
print('孔明和玄德的余弦相似度: ', torch.nn.functional.cosine_similarity(
                    torch.tensor(model.wv.get_vector('孔明')),torch.tensor(model.wv.get_vector('玄德')), dim=0).item())
print('孔明和曹操的余弦相似度: ', torch.nn.functional.cosine_similarity(
                    torch.tensor(model.wv.get_vector('孔明')), torch.tensor(model.wv.get_vector('曹操')), dim=0).item())

# ## 可视化 

# In[159]:


# 将词向量投影到二维空间
rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.index_to_key): #index_to_key 序号,词语
    rawWordVec.append(model.wv[w]) #词向量
    word2ind[w] = i #{词语:序号}
rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec) 


# In[160]:

#降维之前20维
print(rawWordVec)


# In[161]:
#降维之后2维
print(X_reduced)


# In[162]:


# 绘制星空图
# 绘制所有单词向量的二维空间投影
fig = plt.figure(figsize = (15, 10))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.3, color = 'black')


# 绘制几个特殊单词的向量
words = ['孙权', '刘备', '曹操', '周瑜', '诸葛亮', '司马懿','汉献帝']

# 设置中文字体 否则乱码
zhfont1 = matplotlib.font_manager.FontProperties(fname='./micross.ttf', size=16)
for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'orange',markersize=10)
        # plt.text(xy[0], xy[1], w, alpha = 1, color = 'red')
        plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'red')
# plt.show()

# ## 类比关系实验

# In[163]:


# 玄德－孔明＝？－曹操
words = model.wv.most_similar(positive=['玄德', '曹操'], negative=['孔明'])
print(words)


# In[164]:


# 曹操－魏＝？－蜀
words = model.wv.most_similar(positive=['曹操', '蜀'], negative=['魏'])
print(words)

