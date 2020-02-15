
### 分析思路：
>0、数据准备
1、数据探索
2、特征工程
3、建模
4、RFM分析和用户画像
### 0、数据准备
##### 0.1 模块及数据导入
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 解决坐标轴刻度负号乱码
plt.rcParams['axes.unicode_minus'] = False
# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['Simhei']
%matplotlib inline
```
```
df=pd.read_csv(r'D:\Users\wuxiao\Desktop\userlostprob\userlostprob.txt',sep='\t')
df.head()
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-4d1b637cd0c37865.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

###### 0.2 数据项基本信息

![image.png](https://upload-images.jianshu.io/upload_images/18032205-595c4aadec379be8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

数据维度为689945*51，label为标签列，1为未流失，0为流失，samplied是id列。其余有49个特征项，下面将针对部分关键特征进行分析。
- d:预定日期
- arrival:入住日期
- h:访问时间段
- customer_value_profit：客户近1年价值
- ctrip_profits：客户价值
- consuming_capacity：消费能力指数
- price_sensitive:价格敏感指数
- avgprice:入住酒店平均价格
- starprefer:酒店星级偏好
- ordernum_oneyear:年订单数
- ordercanceledprecent:订单取消率
- lasthtlordergap:距离上次预定的时间
- sid:新客老客特征
- hotelcr：酒店cr值
- hoteluv：酒店uv值
- commentnums：酒店点评数
- novoters：酒店点评人数
- cancelrate：酒店取消率
- lowestprice：酒店最低价

![image.png](https://upload-images.jianshu.io/upload_images/18032205-b294e2c7fc3f37fd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

数据存在偏斜，但不平衡程度不大。

![image.png](https://upload-images.jianshu.io/upload_images/18032205-e02e7642db75b069.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

数据缺失值较多，特别是historyvisit_7ordernum缺失达到88%。

### 一、数据探索
##### 1.1 预定日期和入住日期
```
df_d=df.d.value_counts().to_frame().reset_index()
df_arrival=df.arrival.value_counts().to_frame().reset_index()
time_table=df_d.merge(df_arrival,how='outer',on='index')
time_table.fillna(0,inplace=True)
time_table.set_index('index',inplace=True)
time_table.sort_index(inplace=True)

x=time_table.index
y1=time_table.arrival
y2=time_table.d

plt.figure(figsize=(13,5))
plt.style.use('bmh')
plt.plot(x,y1,c="r",label='入住人数');
plt.bar(x,y2,align="center",label='预定人数');
plt.title('访问和入住人数图',fontsize=20)
plt.xticks(rotation=45,fontsize=13)
plt.xlabel('日期');
plt.ylabel('人数',fontsize=13);
plt.legend(fontsize=13)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-a9e62bed2412f436.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 520那天预定人数和入住人数都达到峰值，因为情侣会出门“过节”。521之后入住人数就一路走低。后面有两个小突起是周末。
### 1.2 访问时间段
```
plt.figure(figsize=(15, 6))
plt.hist(df.h.dropna(), bins = 50, edgecolor = 'k');
#因为最多24个时段，所以bins再大的话，只是调整方块的间距了
plt.title('访问时间段',fontsize=20);
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('访问时间',fontsize=18); 
plt.ylabel('人数',fontsize=18); 
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-c490848e53abc377.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 5点是访问人数最少的时点，这个时候大家都在睡觉。5点过后访问人数开始上升，在晚间9、10点的时间段，访问人数是最多的。
##### 1.3
```
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.plot(df.index,df.customer_value_profit,linewidth=0.5)
plt.title('客户近1年价值')

plt.subplot(122)
plt.plot(df.index,df.ctrip_profits,linewidth=0.5)
plt.title('客户价值')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-cdceeb9a514d87b6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 可以看到，“客户近1年价值”和“客户价值”两个特征是非常相关的，都可以用来表示[客户的价值]这么一个特征。同时可以看到，大部分的客户价值都处在0-100这个范围，但是有些客户价值非常大，设置达到了600，这些客户都可以在以后的分析中重点观察，因为他们是非常有“价值”的。

##### 1.4 消费能力指数
```
plt.figure(figsize=(12, 4))

plt.hist(df.consuming_capacity,bins=50,edgecolor='k')
plt.xlabel('消费能力指数')
plt.ylabel('人数')
plt.title('消费能力指数图')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-58809838e342e28a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 可以看到，消费能力指数的值范围是0-100，相当于对酒店客户(及潜在客户)的一个消费能力进行打分。指数值基本呈现一个正态分布的形状，大部分人的消费能力在30附近。当然，我们同时可以看到，消费能力达到近100的人数也非常多，说明在我们酒店的访问和入住客户中，有不在少数的群体是消费水平非常高的，土豪还是多啊。
##### 1.5 价格敏感指数
```
plt.figure(figsize=(12, 4))

plt.hist(df['price_sensitive'].dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('价格敏感指数'); 
plt.ylabel('人数'); 
plt.title('价格敏感指数图');
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-8e84c00540762a3d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 价格敏感指数，用来反映客户对价格的一个在意程度。可以看到，除去两头的极值现象，中间的分布属于右偏的正太分布，大部分人的价格敏感指数比较低，也就是说，大部分客户（及潜在客户）是对价格不是很敏感的，并不会一味地去追求低价的酒店和房间，或许，酒店方面不需要在定价方面花费太多的脑筋。当然，我们也会发现，100处的人数也并不少，还是存在一部分的群体对价格极度敏感的，如果是针对这一部分客户，用一些打折优惠的方式会有意想不到的成效。

##### 1.6 入住酒店平均价格
```
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.hist(df.avgprice.dropna(),bins=50,edgecolor = 'k')
plt.xlabel('酒店价格'); 
plt.ylabel('偏好人数'); 
plt.title('酒店价格偏好');

plt.subplot(122)
plt.hist(df[df.avgprice<2000]['avgprice'].dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('酒店价格'); 
plt.ylabel('偏好人数'); 
plt.title('2000元以内酒店偏好');
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-6e9067d504f3c479.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 从左图中可以知道，酒店平均价格范围为1-6383元，但实际上酒店价格在1000以上的，选择的人就非常少了，价格在2000元以上的酒店就更加是没有人去选择了，所以右图展示了价格为2000元以下的酒店情况。右图表明，消费者对酒店价格的选择，基本是一个正偏态的分布，大部分人会选择的平均价格在300元左右（基本就是7天、如家这类吧）。


##### 1.7 酒店星级偏好
```
plt.figure(figsize=(10, 4))
plt.hist(df.starprefer.dropna(), bins = 50, edgecolor = 'k')
plt.xlabel('星级偏好程度')
plt.ylabel('选择人数');
plt.title('酒店星级偏好')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-20311794206efa79.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 分布有点不规律，尤其是40、60、80、100的分段存在极值情况，剔除这几个分段，星级偏好主要集中在60~80之间。

##### 1.8 用户年订单数

```
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.hist(df.ordernum_oneyear.dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('年订单数'); 
plt.ylabel('人数'); 
plt.title('客户年订单数分布');

plt.subplot(122)
plt.hist(df[df.ordernum_oneyear<100].ordernum_oneyear.dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('年订单数'); 
plt.ylabel('人数'); 
plt.title('年订单数100单内的分布');
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-e7bf0ff85bc9d488.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 用户年订单数最高可达700+，但是大部分用户的你按订单数集中在0~20之间。

##### 1.9 订单取消率
```
plt.figure(figsize=(10, 4))
plt.hist(df.ordercanceledprecent.dropna(),bins=50,edgecolor = 'k')
plt.xlabel('订单取消率')
plt.ylabel('人数')
plt.title('订单取消率')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-43018a5103507d84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 存在大量的用户订单取消率为0，也存在部分极端用户订单取消率为1。

##### 1.10 举例上一次预定的时间
```
plt.figure(figsize=(10, 4))
plt.hist(df.lasthtlordergap.dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('间隔时长'); plt.ylabel('人数'); 
plt.title('距离上次预定的时间');

```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-034bd4cfc3a4893e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 从图像上看，比较符合均值为0的正态分布。但是并不知道间隔时长的单位是什么。
##### 1.11 新老客流失率

```
s_table=df[['label','sid']]
s_table['sid']=np.where(s_table['sid']==1,1,0)
s_table['flag']=1
s=s_table.groupby('sid').sum().reset_index()
s['rate']=s['label']/s['flag']                       # flag求和刚好是sid为0和1的个数，label求和刚好是流失人数，相除则为流失率

plt.figure(figsize=(12, 4))
plt.subplot(121)
percent=[s['flag'][0]/s['flag'].sum(),s['flag'][1]/s['flag'].sum()]
color=['steelblue','lightskyblue']
label=['老客','新访']
plt.pie(percent,autopct='%.2f%%',labels=label,colors=color)
plt.title('新老客户占比')

plt.subplot(122)
plt.bar(s.sid,s.rate,align='center',tick_label=label,edgecolor = 'k')
plt.ylabel('流失率')
plt.title('新老客户中的客户流失率')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-22381da7e8525c4d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 新客大约占5.58%，老客的流失率较新客的流失率更高。
##### 1.12 酒店转换率
```
plt.figure(figsize=(10, 4))
plt.hist(df.hotelcr.dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('酒店cr值');  
plt.title('酒店转换率');
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-1889ce370ecd6a02.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- CR本来是网站转化率（conversion rate）是指用户进行了相应目标行动的访问次数与总访问次数的比率，应当是小于1的数据，但这里酒店的cr处在1~2之间，不知道是怎么定义的。

##### 1.13 酒店独立访客
```
plt.figure(figsize=(10, 4))
plt.hist(df.hoteluv.dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('酒店uv值');  
plt.title('酒店历史独立访客量');
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-a91ab6413c6680bd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- PV(访问量)： 即Page View, 即页面浏览量或点击量，用户每次刷新即被计算一次。UV(独立访客)：即Unique Visitor,访问您网站的一台电脑客户端为一个访客。00:00-24:00内相同的客户端只被计算一次。umm...这里的酒店uv值，不太清楚代表的是什么，还是数据源的问题。。

##### 1.14 当前酒店点评数
```
plt.figure(figsize=(10, 4))
plt.hist(df.commentnums.dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('点评数量');  
plt.title('酒店点评数');
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-5acc46fba535eb40.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 1.15 当前酒店评分人数

```
plt.figure(figsize=(10, 4))
plt.hist(df.novoters.dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('点评人数');  
plt.title('酒店评分人数');
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-16ff1a1ca263e95d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 1.16 当前酒店历史订单取消率
```
plt.figure(figsize=(10, 4))
plt.hist(df.cancelrate.dropna(), bins = 50, edgecolor = 'k');
plt.xlabel('订单取消率');  
plt.title('酒店订单取消率');
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-ac88d1111a63435c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 1.17 当前酒店可订最低价
```
plt.figure(figsize=(10, 4))
plt.plot(df.lowestprice.dropna())
plt.xlabel('酒店最低价');  
plt.title('酒店最低价');
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-5726995d907e2f11.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 二、特征工程
```
# 为了避免在原数据集上进行修改操作，我们将df复制一份
df1=df.copy()
```
##### 2.1 字符串处理
原数据中，arrival和d都是字符串格式，可以将其详见得到“提前预定的天数”，并转化为新的数值特征。
```
## 增加列
# 将两个日期变量由字符串转换为日期型格式
df1['arrival']=pd.to_datetime(df1['arrival'])
df1['d']=pd.to_datetime(df1['d'])
# 生成提前预定时间列
df1['day_advanced']=(df1['arrival']-df1['d']).dt.days

## 删除列
df1=df1.drop(['d','arrival'],axis=1)
```
处理后：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-c9a8731b9acb9e10.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


##### 2.2 异常值处理

把用户价值的两个特征量customer_value_profit、ctrip_profits中的负值按0处理;把delta_price1、delta_price2、lowestprice中的负值按中位数处理。我个人也不知道为什么这么处理，因为里面很多特征并不知道具体含义。

```
filter1=['customer_value_profit','ctrip_profits']
filter2=['delta_price1','delta_price2','lowestprice']

for i in filter1:
    df1.loc[df1[i]<0,i]=0        ##用df1.loc[df1[i]<0][i]=0 会提示无法有点问题，所以还是得用前面的用法
    
for i in filter2:
    temp=df.delta_price1.mean()
    df1.loc[df1[i]<0,i]=temp    ##用df1.loc[df1[i]<0][i]=0 会提示无法有点问题，所以还是得用前面的用法
```
处理后：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-a47b6560e2eb4dc6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 2.3 缺失值处理
原数据中只有iforderpv_24h、sid、h、day_advanced这四个是不存在缺失的，其他的44个特征都是存在缺失值的，并且大部分的缺失值都挺多的，因此，我们接下来需要对缺失值进行处理。
###### 2.3.1 空值删除
首先设定rate为0.2，如果某行或某列的数据缺失率超过(1-0.2)=0.8，则将其删除：
```
# 删除缺失值比例大于80%的行和列
print('删除空值前数据维度是:{}'.format(df1.shape))
df1.dropna(axis=0,thresh=df1.shape[1]*0.2,inplace=True)
df1.dropna(axis=1,thresh=df1.shape[0]*0.2,inplace=True)
print('删除空值后数据维度是:{}'.format(df1.shape))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-49a124a48e52bf73.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 空值删除操作后，样本数据减少了100条，不算多，影响不大；特征值少了一个，进一步查看可以得知，historyvisit_7ordernum这一列被删除了，因为这一列的缺失值比例高达88%，数据缺失过多，我们将其删除。
##### 2.4缺失项补足
数据项基本满足正态分布或者右偏分布，对正态分布项，采用均值填充较合适，对右偏分布项，采用中位数填充更合适。原数据中，businessrate_pre2、cancelrate_pre、businessrate_pre趋于正态分布，齐豫趋于右偏分布（这一点没有详细考证）。

```
filter_mean=['businessrate_pre2','cancelrate_pre','businessrate_pre']
for i in df1.columns:
    if i in filter_mean:
        df1[i].fillna(df1[i].mean(),inplace=True)
    else:
        df1[i].fillna(df1[i].median(),inplace=True)
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-30f58798deef3581.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 2.5 极值处理
有些特征明显有异常大和异常小的值，这里分别用1%和99%分位数替换超过上下限的值。
```
for i in df1.columns:
    df1.loc[df1[i]<np.percentile(df1[i],1),i]=np.percentile(df1[i],1)
    df1.loc[df1[i]>np.percentile(df1[i],99),i]=np.percentile(df1[i],99)
```
##### 2.3.4 相关性分析
```
# 用户特征提取(分两次提取，为了更好地显示图)
user_features=['visitnum_oneyear','starprefer','sid','price_sensitive','ordernum_oneyear','ordercanncelednum','ordercanceledprecent','lastpvgap',
               'lasthtlordergap','landhalfhours','iforderpv_24h','historyvisit_totalordernum','historyvisit_avghotelnum','h',
               'delta_price2','delta_price1','decisionhabit_user','customer_value_profit','ctrip_profits','cr','consuming_capacity','avgprice']
# 生成用户特征的相关性矩阵
corr_mat=df1[user_features].corr()

# 绘制用户特征的相关性矩阵热度图
plt.figure(figsize=(12,12))
sns.heatmap(corr_mat, xticklabels=True, yticklabels=True, square=False, linewidths=.5, annot=True, cmap='Blues')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-baddda4aa8a63b46.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出，不少特征存在强相关性：
- ordernum_oneyear和historyvisit_totalordernum的相关性高达0.93，因为它们都是表示用户1年内的订单数，我们选择其中名字更好识别的ordernum_oneyear作为用户年订单数的特征。
- decisionhabit_user和historyvisit_avghotelnum相关性达到了0.89，说明也是高度相关的，说明可能用户的决策习惯就是根据用户近3个月的日均访问数来设定的，我们可以通过PCA提取一个主成分用来表示用户近期的日均访问量。
- customer_value_profit和ctrip_profits这两个特征之间相关性达到了0.85，这两个特征我们在上面的数据可视化中就有提到，表示的是不同时间长度下衡量的客户价值，必然是高度相关的，我们可以用PCA的方法提取出一个主成分来代表客户价值这么一个信息。
- avgprice和consuming_capacity之间的相关性达到了0.91，同时starprefer与consuming_capacity相关性0.71，starprefer与avgprice相关性0.66，都比较高。这三个特征我们在数据可视化的部分也有提过，它们都代表了消费者的一个消费水平，消费能力越大，愿意或者说是会去选择的酒店的平均价格就会越高，对酒店的星级要求也会越高。可以考虑将这几个变量进行PCA降维。
- delta_price1和delta_price2的相关性高达0.91，同时和avgprice的相关性也大于0.7，针对这几个指标可以抽象出一个指标叫做“用户偏好价格”。
```
# 用户特征提取(分两次提取)
user_features=['hotelcr','hoteluv','commentnums','novoters','cancelrate','lowestprice','cr_pre','uv_pre','uv_pre2','businessrate_pre',
                'businessrate_pre2','customereval_pre2','commentnums_pre','commentnums_pre2','cancelrate_pre','novoters_pre','novoters_pre2',
                'deltaprice_pre2_t1','lowestprice_pre','lowestprice_pre2','firstorder_bu','historyvisit_visit_detailpagenum']
# 生成用户特征的相关性矩阵
corr_mat=df1[user_features].corr()

# 绘制用户特征的相关性矩阵热度图
plt.figure(figsize=(12,12))
sns.heatmap(corr_mat, xticklabels=True, yticklabels=True, square=False, linewidths=.5, annot=True, cmap='Blues')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-51bf024c25c5ea7b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- novoters和commentnums相关性高达0.99，前者是当前点评人数，后者是当前点评数，可以抽象出“酒店热度”指标；
- novoters_pre和commentnums_pre相关性高达0.99，可以抽象出“24小时内浏览次数最多的酒店热度”指标；
- novoters_pre2和commentnums_pre2相关性高达0.99，可以抽象出“24小时内浏览酒店平均热度”指标；
- cancelrate和hoteluv相关性0.76，和commentnums相关性0.84，和novoters相关性0.85，酒店的“人气”高，说明访问的频繁，历史取消率可能也会高一点。
- uv_pre和uv_pre2相关性高达0.9；businessrate_pre和businessrate_pre2相关性高达0.84；commentnums_pre和commentnums_pre2相关性高达0.82；novoters_pre和novoters_pre2相关性高达0.83。这些指标之间都是“浏览最多的酒店的数据”和“浏览酒店的平均数据”的关系，相关性高是正常的，暂时不用抽象出其他的指标。

##### 2.6 降维
```
c_value=['customer_value_profit','ctrip_profits']                   # 用户价值
consume_level=['avgprice','consuming_capacity']                     # 用户消费水平
price_prefer=['delta_price1','delta_price2']                        # 用户偏好价格
hotel_hot=['commentnums','novoters']                                # 酒店热度
hotel_hot_pre=['commentnums_pre','novoters_pre']                    # 24小时内浏览次数最多的酒店热度
hotel_hot_pre2=['commentnums_pre2','novoters_pre2']                 # 24小时内浏览酒店的平均热度

from sklearn.decomposition import PCA
pca=PCA(n_components=1)
df1['c_value']=pca.fit_transform(df1[c_value])
df1['consume_level']=pca.fit_transform(df1[consume_level])
df1['price_prefer']=pca.fit_transform(df1[price_prefer])
df1['hotel_hot']=pca.fit_transform(df1[hotel_hot])
df1['hotel_hot_pre']=pca.fit_transform(df1[hotel_hot_pre])
df1['hotel_hot_pre2']=pca.fit_transform(df1[hotel_hot_pre2])

df1.drop(c_value,axis=1,inplace=True)
df1.drop(consume_level,axis=1,inplace=True)
df1.drop(price_prefer,axis=1,inplace=True)
df1.drop(hotel_hot,axis=1,inplace=True)
df1.drop(hotel_hot_pre,axis=1,inplace=True)
df1.drop(hotel_hot_pre2,axis=1,inplace=True)
df1.drop('historyvisit_totalordernum',axis=1,inplace=True)  ###把重复的一列删了
df1.drop('sampleid',axis=1,inplace=True)   ###把id列删了
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-72cce6afaebac805.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 2.7 数据标准化
```
# 数据标准化
from sklearn.preprocessing import StandardScaler

y=df1['label']
x=df1.drop('label',axis=1)

scaler = StandardScaler()
X= scaler.fit_transform(x)   #先用fit求得训练数据的标准差和均值,再用transform将数据转化成
```

几种标准化的方法：
[https://www.jianshu.com/p/fa73a07cd750](https://www.jianshu.com/p/fa73a07cd750)
考虑到基本都是负荷正态分布或者偏态分布的，所以这里是标准化为标准正态分布，否则的话，采用min-max标准化等其他方法可能会更好。
标准化后的数据形态：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-14c93c1ae4ca73b2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 三、建模
先拆分训练集和数据集
```
from sklearn import model_selection

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size= 0.2,random_state=1)
```

##### 3.1 逻辑回归
```
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

lr = LogisticRegression()                                        # 实例化一个LR模型
lr.fit(X_train,y_train)                                          # 训练模型
y_prob = lr.predict_proba(X_test)[:,1]                           # 预测1类的概率
y_pred = lr.predict(X_test)                                      # 模型对测试集的预测结果
fpr_lr,tpr_lr,threshold_lr = metrics.roc_curve(y_test,y_prob)    # 获取真阳率、伪阳率、阈值
auc_lr = metrics.auc(fpr_lr,tpr_lr)                              # AUC得分
score_lr = metrics.accuracy_score(y_test,y_pred)                 # 模型准确率


print('模型准确率为:{0},AUC得分为:{1}'.format(score_lr,auc_lr))
print('  ')
print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-ab36eb60c7d97754.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 此时的模型准确率是以分为0类、1类的概率大小进行分类的，所以理论上通过调整分类阈值，可以达到更高的精度。
- 在不对阈值进行调整情况下，从混淆矩阵中可以看出，1类的recall偏小，表明更容易被分为0类，这种情况对应的是ROC曲线中的左下方低点，分为0类的阈值应该调大，分为1类的阈值应该调低。


##### 3.2 朴素贝叶斯

```
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report

gnb = GaussianNB()                                                # 实例化一个LR模型
gnb.fit(X_train,y_train)                                          # 训练模型
y_prob = gnb.predict_proba(X_test)[:,1]                           # 预测1类的概率
y_pred = gnb.predict(X_test)                                      # 模型对测试集的预测结果
fpr_gnb,tpr_gnb,threshold_gnb = metrics.roc_curve(y_test,y_prob)    # 获取真阳率、伪阳率、阈值
auc_gnb = metrics.auc(fpr_gnb,tpr_gnb)                              # AUC得分
score_gnb = metrics.accuracy_score(y_test,y_pred)                 # 模型准确率


print('模型准确率为:{0},AUC得分为:{1}'.format(score_gnb,auc_gnb))
print('  ')
print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-83ebd280ee01b647.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

查看了一下预测的分为1类和0类的概率：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-076438d2a2d8c774.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

所以这里的贝叶斯概率并不是真实概率（真实概率为所有独立变量概率的成绩，理论上是一个很小很小的值，而不会是一个mean值在0.5左右的值）。

##### 3.3 支持向量机
```
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report

svc = SVC(kernel='rbf',C=1,max_iter=100).fit(X_train,y_train)
y_prob = svc.decision_function(X_test)                              # 决策边界距离
y_pred = svc.predict(X_test)                                        # 模型对测试集的预测结果
fpr_svc,tpr_svc,threshold_svc = metrics.roc_curve(y_test,y_prob)     # 获取真阳率、伪阳率、阈值
auc_svc = metrics.auc(fpr_svc,tpr_svc)                              # 模型准确率
score_svc = metrics.accuracy_score(y_test,y_pred)

print('模型准确率为:{0},AUC得分为:{1}'.format(score_gnb,auc_gnb))
print('  ')
print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-8c8c223aef4ab543.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 不设置max_iter会陷入死循环，说明一直无法找到最优解平面。而且无论是rbf核还是多项式核，整体的预测精度都很低。
##### 3.4 决策树
```
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report

dtc = tree.DecisionTreeClassifier()                              # 建立决策树模型
dtc.fit(X_train,y_train)                                         # 训练模型
y_prob = dtc.predict_proba(X_test)[:,1]                          # 预测1类的概率
y_pred = dtc.predict(X_test)                                     # 模型对测试集的预测结果 
fpr_dtc,tpr_dtc,threshod_dtc= metrics.roc_curve(y_test,y_prob)   # 获取真阳率、伪阳率、阈值               
auc_dtc = metrics.auc(fpr_dtc,tpr_dtc)                           # AUC得分
score_dtc = metrics.accuracy_score(y_test,y_pred)                # 模型准确率

print('模型准确率为:{0},AUC得分为:{1}'.format(score_dtc,auc_dtc))
print('  ')
print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-2ec6c81b25d3f4f6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 3.5 随机森林
```
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

rfc = RandomForestClassifier()                                     # 建立随机森林分类器
rfc.fit(X_train,y_train)                                           # 训练随机森林模型
y_prob = rfc.predict_proba(X_test)[:,1]                            # 预测1类的概率
y_pred=rfc.predict(X_test)                                         # 模型对测试集的预测结果
fpr_rfc,tpr_rfc,threshold_rfc = metrics.roc_curve(y_test,y_prob)   # 获取真阳率、伪阳率、阈值  
auc_rfc = metrics.auc(fpr_rfc,tpr_rfc)                             # AUC得分
score_rfc = metrics.accuracy_score(y_test,y_pred)                  # 模型准确率

print('模型准确率为:{0},AUC得分为:{1}'.format(score_rfc,auc_rfc))
print('  ')
print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-bd9e4edf1bf80dbb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 3.6 模型比较
```
plt.style.use('bmh')
plt.figure(figsize=(13,10))

plt.plot(fpr_lr,tpr_lr,label='lr: {0:.3f}'.format(score_lr))                             # 逻辑回归
plt.plot(fpr_gnb,tpr_gnb,label='gnb:{0:.3f}'.format(score_gnb))                          # 朴素贝叶斯模型
plt.plot(fpr_svc,tpr_svc,label='svc:{0:.3f}'.format(score_svc))                                             # 支持向量机模型
plt.plot(fpr_dtc,tpr_dtc,label='dtc:{0:.3f}'.format(score_dtc))                          # 决策树
plt.plot(fpr_rfc,tpr_rfc,label='rfc:{0:.3f}'.format(score_rfc))                          # 随机森林

plt.legend(loc='lower right',prop={'size':25})
plt.xlabel('误诊率')
plt.ylabel('灵敏度')
plt.title('ROC曲线')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-5094238466f4064e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

随机森林胜出！

### 四、RFM分析和用户画像
##### 4.1 RFM分析
RFM模型，即为：
R(Rencency):最近一次消费
F(Frequency):消费频率
M(Monetary):消费金额

![image.png](https://upload-images.jianshu.io/upload_images/18032205-3f56df91f2d1520e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在本案例中，我们选择lasthtlordergap（距离上次下单的时长）、ordernum_oneyear（用户年订单数）、consume_level（用户消费水平）分别作为R、F、M的值，对我们的用户群体进行聚类。
```
rfm = df1[['lasthtlordergap','ordernum_oneyear','consume_level']]  #consume_level是PCA后的特征变量

#归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(rfm)
rfm = pd.DataFrame(scaler.transform(rfm),columns=['recency','frequency','monetary'])

# 分箱
rfm['R']=pd.qcut(rfm["recency"], 2)
rfm['F']=pd.qcut(rfm["frequency"], 2)
rfm['M']=pd.qcut(rfm["monetary"], 2)

# 根据分箱情况编码
from sklearn.preprocessing import LabelEncoder

#从0开始编码的，所以这里直接编码是可以的
rfm['R']=LabelEncoder().fit(rfm['R']).transform(rfm['R'])      #这里需要注意，R为距离上次下单的市场，越小则代表价值越高，所以这一点是反的
rfm['F']=LabelEncoder().fit(rfm['F']).transform(rfm['F'])
rfm['M']=LabelEncoder().fit(rfm['M']).transform(rfm['M'])

def get_label(r,f,m):
    if (r==0)&(f==1)&(m==1):
        return '高价值客户'
    if (r==1)&(f==1)&(m==1):
        return '重点保持客户'
    if((r==0)&(f==0)&(m==1)):
        return '重点发展客户'
    if (r==1)&(f==0)&(m==1):
        return '重点挽留客户'
    if (r==0)&(f==1)&(m==0):
        return '一般价值客户'
    if (r==1)&(f==1)&(m==0):
        return '一般保持客户'
    if (r==0)&(f==0)&(m==0):
        return '一般发展客户'
    if (r==1)&(f==0)&(m==0):
        return '潜在客户'

def RFM_convert(df):
    df['Label of Customer']=df.apply(lambda x:get_label(x['R'],x['F'],x['M']),axis=1)
    
    df['R']=np.where(df['R']==0,'高','低')
    df['F']=np.where(df['F']==1,'高','低')
    df['M']=np.where(df['M']==1,'高','低')
    
    return df[['R','F','M','Label of Customer']]

rfm0=RFM_convert(rfm)
rfm0.head(10)
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-212bd644f7dcb203.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

各类客户的占比：
```
temp=rfm0.groupby('Label of Customer').size()

plt.figure(figsize=(12,12))
colors=['deepskyblue','steelblue','lightskyblue','aliceblue','skyblue','cadetblue','cornflowerblue','dodgerblue']
plt.pie(temp,radius=1,autopct='%.1f%%',pctdistance=0.75,colors=colors)
plt.pie([1],radius=0.6,colors='w')   ##可以用这种方式画空心
plt.title('客户细分情况')
plt.legend(temp.index)
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-4c3430dca4847861.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 潜在客户占比达12。3%，这类客户是rmf指标均不是很好的客户，有待开发；
- 高价值客户11%,重点保持客户10.1%，重点发展客户7%，这是要重点关注的客户群体。

##### 4.2 用户画像
其实我们并不想将用户分的这么细，并且我们其实有挺多的用户行为特征数据，所以也并不想仅用RFM这3个指标进行分析。所以，我们接下来用K-Means聚类的方法将用户分为3类，观察不同类别客户的特征。
```
# 选取出几个刻画用户的重要指标
user_feature = ['decisionhabit_user','ordercanncelednum','ordercanceledprecent','consume_level','starprefer','lasthtlordergap','lastpvgap','h','sid',
                'c_value','landhalfhours','price_sensitive','price_prefer','day_advanced','historyvisit_avghotelnum','ordernum_oneyear']
user_attributes = df1[user_feature]
user_attributes.head()

# 数据标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(user_attributes)

user_attributes = scaler.transform(user_attributes)
```
```
from sklearn.cluster import KMeans

Kmeans=KMeans(n_clusters=3)                                                     # 建立KMean模型
Kmeans.fit(user_attributes)                                                     # 训练模型
k_char=Kmeans.cluster_centers_                                                  # 得到每个分类的质心
personas=pd.DataFrame(k_char.T,index=user_feature,columns=['0类','1类','2类'])  # 用户画像表

plt.figure(figsize=(5,10))
sns.heatmap(personas, xticklabels=True, yticklabels=True, square=False, linewidths=.5, annot=True, cmap='Blues')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-661b96e2dc5847fb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 2类用户的R（lasthtlordergap）为-0.17非常小（R越小越好，这里是反的），F（ordernum_oneyear）为1.1比较高了，M（consume_level）为1.3也几乎是最高的。很明显，2类客户为我们的“高价值客户”；而0类中几乎都是白格子，无论是客户价值还是消费水平值都是最低的，很明显，这一类我们将其归为“低价值客户”；剩下的1类我们将其称为“中等群体”。
```
plt.figure(figsize=(9,9))

class_k=list(Kmeans.labels_)                          # 每个类别的用户个数
percent=[class_k.count(1)/len(user_attributes),class_k.count(0)/len(user_attributes),class_k.count(2)/len(user_attributes)]   # 每个类别用户个数占比

fig, ax = plt.subplots(figsize=(10,10))
colors=['aliceblue','steelblue','lightskyblue']
types=['中等群体','低价值用户','高价值用户']
ax.pie(percent,radius=1,autopct='%.2f%%',pctdistance=0.75,colors=colors,labels=types)
ax.pie([1], radius=0.6,colors='w')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-3e9da431f7a96d91.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 可以看到，“低价值客户”的占比非常之大，中等人群占比最小。
##### 4.3 用户画像分析
- 高价值用户分析（2类用户）
消费水平高，客户价值大，追求高品质，对酒店星级要求高，访问频率和预定频率都较高，提前预定的时间都较短，决策一般都较快（日均访问数少），订单取消率较高，可以分析出这类客户商务属性偏重，可能随时要出差，因此都不会提前预定，可能出差随时会取消，因此酒店取消率也会更高一点。sid的值较大，说明高价值客户群体多集中在老客户中。价格敏感度较高，说明可能比较要求性价比。h值非常小，可能访问和预定时间多在半夜或是清晨。
这部分客户对于我们而言是非常重要的，因此我们需要对其实施个性化的营销：
1、为客户提供更多差旅酒店信息。
2、多推荐口碑好、性价比高的商务酒店。
3、推荐时间集中在半夜或是清晨。
- 中等价值用户分析（1类用户）
消费水平和客户价值都偏低，对酒店品质也不太追求，访问和预定频率也都较高，提前预定的时间是三类中最长的，最值得注意的是，0类客户中有两个颜色非常深的蓝色格子，是用户决策和近3个月的日均访问数。可以看出，这类客户通常很喜欢逛酒店界面，在决定要订哪家酒店前通常会花费非常多的时间进行浏览才能做出选择，并且一般都会提前很久订好房。我们可以给这类客户打上“谨慎”的标签。我们可以合理推断，这一类客户，可能预定酒店的目的多为出门旅行。
针对这部分客户，我们需要：
1、尽可能多地进行推送，因为此类客户通常比较喜欢浏览。
2、推送当地旅游资讯，因为这类客户旅游出行的概率较大。
3、多推荐价格相对实惠的酒店。
- 低价值用户分析（0类用户）
消费水平和客户价值极低，对酒店品质不追求，偏好价格较低，决策时间很短，访问和预定频率很低，sid值很低，说明新客户居多。
针对这部分客户，我们需要：
1、不建议花费过多营销成本，但因为新用户居多，属于潜在客户，可以维持服务推送。
2、推送的内容应多为大减价、大酬宾、跳楼价之类的。
3、此类用户占比居多，可进一步进行下沉分析，开拓新的市场。
