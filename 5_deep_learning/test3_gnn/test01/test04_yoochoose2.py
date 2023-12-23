#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/23 14:29
# Module    : test04_yoochoose.py
# explain   : 电商购买预测

import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# 读取数据

# 用户的点击行为数据
df = pd.read_csv('../../data/yoochoose/yoochoose-clicks.dat', header=None)
df.columns = ['session_id', 'timestamp', 'item_id', 'category']
# 用户有没有购买商品
buy_df = pd.read_csv('../../data/yoochoose/yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']


item_encoder = LabelEncoder()
df['item_id'] = item_encoder.fit_transform(df.item_id)

"""
session_id相同代表是同一个人, 点了四个网页----某一个人的点击行为
item_id:代表东西是什么(商品id号)
"""
print(df.head())
print(buy_df.head())


# 数据量有点多，这里我们只选择其中一小部分--100000--条来建模。
#数据有点多，咱们只选择其中一小部分来建模
#unique：唯一性索引
#选择十万条来建模
sampled_session_id = np.random.choice(df.session_id.unique(), 100000, replace=False)
df = df.loc[df.session_id.isin(sampled_session_id)]
df.nunique()

# 另外，把标签也拿到手。取标签需要跟yoochoose-buys.dat数据表做关联。
df['label'] = df.session_id.isin(buy_df.session_id)
# 得到标签label, True或False，表示有没有购买数据。
print(df.head())

'''
制作数据集。 制作成传入pytorch_geometric需要的数据形式。 
这里需要注意以下几个方面的内容：
        ①首选，我们需要把每一个session_id(代表一个用户登录)都当做一个图，其中每一个图都具有多个点和一个标签。
        ②其中每个图中的点就是其item_id，特征暂且用其id来表示，之后会做embedding。
        ③这里的任务有点类似与NLP中的任务，在NLP任务中，拿到词之后会先把词转换成对应的id，然后做embedding(查询做好的词向量表)。用户的点击顺序是不会调换换的。

数据集制作流程：
        ①首先遍历数据中每一组session_id，目的是将其制作成pytorch_geometric格式。
        ②对每一组session_id中的所有item_id进行编码(图中点的索引)，从0开始，按数值大小进行编码。例如(46,1653,372,5768)--->(0,2,1,3)。
        ③这样编码的目的是制作邻接矩阵edge_index。edge_index需要从0,1,2,3...开始。
        ④浏览是有顺序的，浏览顺序从source_nodes到target_nodes，比如(0,0,2,1)，则source_nodes:[ 0 0 2],target_nodes[0 2 1]。
        ⑤data = Data(x=x, edge_index=edge_index, y=y)。
        ⑥最后将数据集保存下来（以后就不用重复处理了）。
原文链接：https://blog.csdn.net/weixin_42254289/article/details/131291198
'''
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property  # python装饰器， 只读属性，方法可以像属性一样访问
    def raw_file_names(self):  # ①检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # ②如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    def processed_file_names(self):  # ③检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，有则直接加载
        # ④没有就会走process,得到'yoochoose_click_binary_1M_sess.dataset'文件
        return ['yoochoose_click_binary_1M_sess.dataset']

    def download(self):  # ①检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # ②如有文件不存在，则调用download()方法执行原始文件下载
        pass

    def process(self):  # ④没有就会走process,得到'yoochoose_click_binary_1M_sess.dataset'文件

        data_list = []  # 保存最终生成图的结果

        # process by session_id
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features

            y = torch.FloatTensor([group.label.values[0]])
            # 创建图
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)  # 转换成可以保存到本地的格式
        torch.save((data, slices), self.processed_paths[0])  # 保存操作，名字跟yoochoose_click_binary_1M_sess.dataset一致


from tqdm import tqdm  # 进度条

df_test = df[:100]  # 取前100个
grouped = df_test.groupby('session_id')  # 基于session_id分组
for session_id, group in tqdm(grouped):  # 遍历每一组的session_id，都做成一个图
    print('session_id:', session_id)
    # LabelEncoder：sklearn中的包,对数值做转换
    sess_item_id = LabelEncoder().fit_transform(group.item_id)  # 把item_id做一个转换，转换成从0开始的格式，赋值给sess_item_id
    print('sess_item_id:', sess_item_id)
    group = group.reset_index(drop=True)  # 重置索引
    group['sess_item_id'] = sess_item_id
    print('group:', group)
    # 设置点的标签为item_id    drop_duplicates:去除重复项的操作
    node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
        'sess_item_id').item_id.drop_duplicates().values
    print('node_features:', node_features)
    node_features = torch.LongTensor(node_features).unsqueeze(1)  # unsqueeze:指定的位置插入一个维度
    print('node_features:', node_features)
    print('node_features:', node_features.shape)  # torch.Size([3, 1])

    # 因为是顺序结构，所以邻接矩阵可以通过这种方式构建
    target_nodes = group.sess_item_id.values[1:]  # 取出target
    source_nodes = group.sess_item_id.values[:-1]  # 取出source
    print('target_nodes:', target_nodes)
    print('source_nodes:', source_nodes)
    # 指定边索引
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    x = node_features
    y = torch.FloatTensor([group.label.values[0]])
    print(f"y:{y}")
    data = Data(x=x, edge_index=edge_index, y=y)
    print('data:', data)



# 其中TopKPooling类似于下采样，是剪枝的过程，选择得分比较低的节点剪枝掉，然后再重新组合成一个新的图。

embed_dim = 128
from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class Net(torch.nn.Module):  # 针对图进行分类任务
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(embed_dim, 128)  # 卷积层 输入embed_dim，输出128
        self.pool1 = TopKPooling(128, ratio=0.8)  # 做剪枝操作
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() + 10, embedding_dim=embed_dim)  # 映射向量
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的
        # print(x)
        x = self.item_embedding(x)  # n*1*128 特征编码后的结果
        # print('item_embedding',x.shape)
        x = x.squeeze(1)  # n*128
        # print('squeeze',x.shape)

        """
        对输入不断做卷积，不断做池化池化，得到的特征会越来越浓缩，图会越来越小，
        但是池化完成之后的特征维度都是一样的

        """
        x = F.relu(self.conv1(x, edge_index))  # n*128
        # print('conv1',x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        # print('self.pool1',x.shape)
        # print('self.pool1',edge_index)
        # print('self.pool1',batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(x, batch)  # gap:全局平均池化  得到全局特征
        # print('gmp',gmp(x, batch).shape) # batch*128
        # print('cat',x1.shape) # batch*256
        x = F.relu(self.conv2(x, edge_index))
        # print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # print('pool2',x.shape)
        # print('pool2',edge_index)
        # print('pool2',batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = gap(x, batch)
        # print('x2',x2.shape)
        x = F.relu(self.conv3(x, edge_index))
        # print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # print('pool3',x.shape)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        # print('x3',x3.shape)# batch * 256
        x = x1 + x2 + x3  # 获取不同尺度的全局特征
        """通过全连接层，得到最终输出结果值"""
        x = self.lin1(x)
        # print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)
        # print('lin2',x.shape)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)  # batch个结果
        # print('sigmoid',x.shape)
        return x


# 模型的训练和评估
from torch_geometric.loader import DataLoader


def train():
    model.train()

    loss_all = 0
    for data in train_loader:  # 遍历dataloader
        data = data
        # print('data',data)
        optimizer.zero_grad()
        output = model(data)  # data数据传入模型
        label = data.y
        loss = crit(output, label)  # 计算损失
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()  # 梯度更新
    return loss_all / len(dataset)


model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()
train_loader = DataLoader(dataset, batch_size=64)
for epoch in range(10):
    print('epoch:', epoch)
    loss = train()
    print(loss)

from sklearn.metrics import roc_auc_score


def evalute(loader, model):
    model.eval()

    prediction = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data  # .to(device)
            pred = model(data)  # .detach().cpu().numpy()

            label = data.y  # .detach().cpu().numpy()
            prediction.append(pred)
            labels.append(label)
    prediction = np.hstack(prediction)
    labels = np.hstack(labels)

    return roc_auc_score(labels, prediction)


for epoch in range(1):
    roc_auc_score = evalute(dataset, model)
    print('roc_auc_score', roc_auc_score)

