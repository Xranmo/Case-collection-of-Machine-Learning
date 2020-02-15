**数据来源kaggle(uci数据集)： [https://www.kaggle.com/janiobachmann/bank-marketing-dataset/kernels]**

>目录： 
0 项目概述 
一、业务分析 
　　1.1 基本属性 
　　1.2 业务联系
　　1.3 最近一次营销活动
　　1.4 目标数据
  
  
二、数据准备 
　　2.1 描述性数据概览 
　　2.2 数据清洗和过滤
三、探索性数据分析（EDA）
　　3.1 数据项分布
　　　　3.1.1 盈余
　　　　3.1.2 职业
　　　　3.1.3 婚姻状态
　　　　3.1.4 受教育程度
　　　　3.1.5 有无住房贷款和个人贷款
　　3.2 是否有定期存款？
四 多重探究 
　　4.1 营销活动开展的月份 
　　4.2 潜在客户的年龄 
　　4.3 目标群体的职业分析
五、影响客户定期存款业务的特征相关性分析 
　　5.1 矩阵相关性分析 
　　5.2 住房贷款和个人贷款 
六、分类模型 
　　6.1 模型概述 
　　　　6.1.1 模型目标
　　　　6.1.2 建模过程
　　6.1.3 决策树
　　6.1.4 贝叶斯/费舍尔分类
　　6.1.5 神经网络
　　6.1.6 SVM
　　6.1.7 确定最佳模型
七、营销建议
　　7.1 营销目标客户群体
　　7.2 营销策略**

### 0 项目概述 
&emsp;&emsp;本项目的目的是充分挖掘客户的需求、刻画客户群体肖像，并针对营销活动的开展提供建设性的意见建议，从而真正促进推动银行业务的开展。 为此，我们需要对以下加点进行深入挖掘： 
（1）目标人群：哪一部分人群是精准营销的客户群体，针对这一部分人开展营销推广，将使得活动变得高效、快速； 
（2）营销渠道：有哪些营销渠道可以采用，例如电话、电视、社交媒体等，如何针对人群设定最佳的渠道策略； 
（3）定价：具体的业务应该怎样定价以吸引客户？ 
（4）营销策略：推动业务落地，从而真正推动业务实效化开展。
&emsp;&emsp;本数据集的营销场景是给客户推荐定期存款业务。
### 一、业务分析 
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
bank=pd.read_csv('/Users/ranmo//Desktop/数据分析案例/银行营销/bank.csv')
bank.info()
bank.head()
```
显示如下：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-6b9fe292ca213de0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/18032205-17b2e40563d6376f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

一共11162行*17列数据，具体的数据项可以分为三个部分。
##### 1.1 基本属性 
- age,年龄
- job，工作类型（分类：'管理员'，'蓝领'，'企业家'，'女佣'，'管理层'，'退休'，'自雇'，'服务'，'学生' '技术人员'， '失业'， '未知'）
- marital，婚姻状态（分类：'离婚'，'已婚'，'单身'，'未知';注：'离婚'是指离婚或丧偶）
- education，教育程度（分类：'初等教育'，'中等教育'，'高等教育'）
- default，有无违约（分类：'无'，'有'，'未知'）
- housing，有无住房贷款（分类：'无'，'有'，'未知'）
- load：有无个人贷款（分类：'无'，'有'，'未知'）
- balance：盈余（收支平衡）
##### 1.2 业务联系
- contact：联系方式（分类：'移动电话'，'座机'）
- day:上一个联系日（分类：'周一'，'周二'，'周三'，'周四'，'周五'）
- month：上一个联系月（分类：'一月~十二月'）
- duration：通话时间，秒（此数据为通话时间，似乎包含等待接通的时间，因为最小值是2s，之后会分析到）
##### 1.3 最近一次营销活动
- campain：上一次营销活动和此客户联系的次数
- pdays：自上一次营销活动联系后，至今的天数
- previous：上一次营销活动之前和客户累计联系过的次数
- poutcome：上一次营销的结果（分类：'失败'，'未知'，'其他'，成功'）
##### 1.4 目标数据
- deposit：客户是否有定期存款？（分类：'是'，'否'）

### 二、数据准备 
##### 2.1 描述性数据概览 

![image.png](https://upload-images.jianshu.io/upload_images/18032205-849adb21a4820f66.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得到以下基本信息：
- 客户平均年龄约为41岁，最高为95岁， 最低为18岁；
- 客户平均盈余为1528，但标准差很大，说明此项数据的分布情况很分散。
- 通话持续时间在2~3881s（1h+）不等，是上一次的通话时间还是累计的通话时间？？是纯通话时间还是包含等待时间？这项数据不敢轻易使用。不过可以确定的是，通话时间越长，肯定说明客户潜力越大，相应的存款也会更多。
- 上一次营销活动的联系次数在1~63次不等，相应的联系次数越多，则约表明该客户在上一次活动中参与度高；
- 上一次营销活动后至今的天数为-1~854天，为什么会有-1？是否为数据错误；
- 上一次营销活动之前和客户累计联系过的次数为0~58次，整体数据偏小。
##### 2.2 数据清洗和过滤

![image.png](https://upload-images.jianshu.io/upload_images/18032205-0593ee5dc57854bb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因为没有缺失数据，所以不用进行数据填充。
针对pdays中存在的“-1”，也没有进行清洗。
### 三、探索性数据分析（EDA）
##### 3.1 数据项分布
是否有定期存款是我们特别关注的数据，不过在此之前，我们可以先分析一下各个数据项的分布以及彼此可能存在的联系。
```
bank.hist(bins=20,figsize=(14,10))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-7dc650f4e975c5cb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

###### 3.1.1 盈余
和是否有违约之间的关系:
```
sns.set(style="darkgrid")
sns.boxplot(x='default',y='balance',hue='deposit',data=bank)
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-f581a305207a1405.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

和职业之间的关系：
```
sns.boxplot(x='job',y='balance',hue='deposit',data=bank)
plt.xticks(rotation=90)
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-0317239cc1534d2f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

和教育程度之间的关系:
```
sns.violinplot(x='education',y='balance',hue='deposit',data=bank)
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-a716ee17b1cbc38b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得知：
- 有违约记录的人员盈余明显偏低，表明他们的经济状况确实不太好；
- 有几个职业经济状况更好，退休人员、管理层、自雇和技术人员；
- 不同教育程度的人员的盈余情况似乎没有明显的偏差，并不像我们想象中的，高等教育者应当具备更高的盈余。
###### 3.1.2 职业
职业的数量分布：
```
plt.rcParams['figure.figsize']=(10,6)
sns.set()
sns.barplot(x='index',y='job',data=bank['job'].value_counts().to_frame().reset_index())
plt.xticks(rotation=90)
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-c2740f23e1257682.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

职业和年龄的关系：
```
sns.boxplot(x='job',y='age',data=bank)
plt.xticks(rotation=90)
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-9bec333328cd793b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

职业和收入的关系：（发现画图效率有点低，已经开始用tableau混用了）
```
#tableau创建计算字段balance status：
if [balance]<0
then 'negtive'
elseif [balance]<3000
then 'low'
elseif [balance]<10000
then 'mid'
else 'high'
end
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-943481dd0878b2e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得知：
- 管理人员是最为普遍的职业类型；
- 退休人员的年龄偏高，而学生最低，umm，跟预期的一致；
- 管理人员和技术人员是综合盈余最高的人（含有比较多的high——balance比例和数目）。
###### 3.1.3 婚姻状态
婚姻状态和盈余的关系：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-f0874e258f0d2374.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得知：
- 似乎婚姻状态与盈余没有什么相关性，因为无论是离婚者、单身者、结婚者，在各个盈余段上的分布都比较相似，普遍分布在0~5k内。
###### 3.1.4 受教育程度
受教育程度和婚姻状态的关系：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-1b3e1451e62dd600.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

受教育程度和盈余的关系（这里是求的各种教育程度中negtive、low、mid、high的中值）：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-77bab09650c41a53.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得知：
- 不同教育程度的离婚率相差不大，但是教育程度越高，单身率越高；
- 不同教育程度的盈余状况似乎相差不大，因为集中盈余状态的中值都差不多，包括high_balance状态下的种植情况也差不多（不考虑unknow）
###### 3.1.5 有无住房贷款和个人贷款
和盈余的关系：
```
plt.rcParams['figure.figsize']=(20,10)
plt.subplot(121)
sns.stripplot(x='housing',y='balance',data=bank)
plt.subplot(122)
sns.stripplot(x='loan',y='balance',data=bank)
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-1c399c1e95e0d0ef.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得知：
- 有无住房贷款和个人贷款会直接影响盈余，没有住房贷款和个人贷款的将具有更多的盈余。
##### 3.2 是否有定期存款？
有无定期存款是我们最为关心的问题，也是直接影响预测模型精度的关键参数，首先我们可以进行整体的比例分析：
```
plt.rcParams['figure.figsize']=(10,6)
f, ax = plt.subplots(1,2)
plt.suptitle('Information on Term Suscriptions', fontsize=20)
bank["deposit"].value_counts().plot.pie(ax=ax[0],autopct='%.2f%%',explode=[0,0.25],startangle=25)
sns.barplot(x='education',y='balance',hue='deposit',data=bank,estimator=lambda x: len(x) / len(bank) * 100)
ax[1].set(ylabel='(%)')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-ec502148a37e3bd7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里我们似乎可以粗略地得知，受教育程度越高（tertiary），越趋于拥有定期存款，具体的存款业务与各个特征变量之间的关系将在第五章详细分析。
### 四 多重探究 
##### 4.1 营销活动开展的月份 

```
import datetime
# date=bank.pdays
now=datetime.datetime.today()
bank_date=bank
bank_date['compain_date']=bank_date.pdays.transform(lambda x:now-datetime.timedelta(days=x))
bank_date['month']=bank_date['compain_date'].transform(lambda x:x.strftime('%m'))
plt.bar(bank_date['month'].value_counts().index,bank_date['month'].value_counts())
plt.xlabel('month')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-1b97e9138524806a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
data=bank_date.groupby(['month','poutcome']).count().reset_index()
sns.barplot(x='month',y='age',data=data,hue='poutcome')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-5a410ac230c498c8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

尽管8月份的营销客户数很多，但是营销结果却存在大量未知的数据。去除这部分数据进行分析：
```
sns.barplot(x='month',y='age',data=data[data['poutcome']!='unknown'],hue='poutcome')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-8fe6a8a0251fac97.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得知：
- 营销活动主要集中在8月、5月、2月；
- 实际营销活动较为成功的月份为5月、2月，当然也很有可能是因为营销的次数比较多导致的成功案例增多，而由于8月的营销结果存在大量未知数据因此无法具体分析。
##### 4.2 潜在客户的年龄 
```
plt.subplot(211)
sns.distplot(bank[bank.deposit=='yes'].age)
plt.ylabel('deposit=yes')
plt.subplot(212)
sns.distplot(bank[bank.deposit=='no'].age)
plt.ylabel('deposit=no')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-bebb95bc33316434.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
data=bank
data['age_status']=data['age']
data.age_status=data.age_status
def agerank(age):
    if age<20:
        age_status='teen'
    elif age>=20 and age<20:
        age_status='young'
    elif age>=30 and age<40:
        age_status='mid'
    elif age>=40 and age<60:
        age_status='mid_old'
    else:age_status='old'
    return age_status
data.age_status=data.age_status.transform(lambda x:agerank(x))
data2=(data.groupby(['age_status','deposit']).age.count()/data.groupby(['age_status']).age.count()).to_frame().reset_index()
sns.barplot(x='age_status',y='age',data=data2,hue='deposit')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-1efab94c81d8f4fe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
data3=(data[data.poutcome!='unknown'].groupby(['age_status','poutcome']).age.count()/data.groupby(['age_status']).age.count()).to_frame().reset_index()
sns.barplot(x='age_status',y='age',data=data3,hue='poutcome')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-fefd42a91d0b92d4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得知：
- 20岁以下或60岁以上的群体会更趋于拥有定期存款；
- 从营销结果来看，20岁以下或60岁以上的群体似乎更容易营销成功，因此可以考虑将他们作为重点的营销对象。
##### 4.3 目标群体的职业分析
```
data['percent']=1
data4=(data.groupby(['job','deposit']).percent.count()/data.groupby(['job']).percent.count()).to_frame().reset_index()
data5=(data[data.poutcome!='unknown'].groupby(['job','poutcome']).percent.count()/data.groupby(['job']).percent.count()).to_frame().reset_index()

plt.subplot(211)
sns.barplot(x='job',y='percent',data=data4,hue='deposit')
plt.subplot(212)
sns.barplot(x='job',y='percent',data=data5,hue='poutcome')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-3ac499195d31cc15.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得出：
- 学生及退休人员更加可能有定期存款，同时在营销方面取得成功；
- 蓝领、企业家、服务者、技术员不容易推销成功。
### 五、影响客户定期存款业务的特征相关性分析 
根据前文的分析，我们初步知道：
（1）age，小于20及大于60岁更趋于拥有定期存款
（2）job，学生和退休者更趋于拥有定期存款
（3）marital，婚姻状态似乎与业务没有太大联系
（4）education，受教育程度越高（tertiary），越趋于拥有定期存款
（5）default，有无违约似乎与业务没有太大联系
（6）housing，尚未分析
（7）load，尚未分析
（8）balance，盈余状态似乎与业务没有太大联系
（9）contact，无关变量
（10）day，尚未分析
（11）month，尚未分析
（12）duration，尚未分析
（13）compain，尚未分析
（14）pdays，尚未分析
（15）poutcome，尚未分析
未分析的几个特征变量中，有的是数值型变量，有的是字符串变量，数值变量采用矩阵相关性分析，其余的进行特性分析。
##### 5.1 矩阵相关性分析 
```
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
data5=bank
data5['deposit']=LabelEncoder().fit_transform(data5['deposit'])
#把deposit转化为数值变量
corrmat=data5.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corrmat,annot=True,cmap=sns.diverging_palette(220, 20, as_cmap=True))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-c617788cb9a60d3e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出，这几个值当中与业务最为相关的就是duration通话时间了，进一步分析：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-de66dc3b4a6e95af.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出，duration的值主要集中在0~600之间，随着duration越大，开设定期存款的比例越高。
```
data5['duration_status']=(data5.duration-data5.duration.mean())
def dur_status(duration_status):
    if duration_status>=0:
        a='above_average'
    else:a='below_average'
    return a
data5['duration_status']=data5['duration_status'].transform(lambda x:dur_status(x))
percentage=(data5.groupby(['duration_status','deposit']).duration.count()/data5.groupby(['duration_status']).duration.count()).to_frame().reset_index()
percentage['percent']=percentage.duration
sns.barplot(x='duration_status',y='percent',data=percentage,hue='deposit',)

```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-fcd95fa52eac61ec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通话时间的均值为375s，以此作为分解，可以看到通话时间高于均值的，开办业务的比例为77.3%，低于均值的，仅为31.6%。
可以得知：
- 随着通话时间越长，表明用户开办业务的成功率越高。
##### 5.2 住房贷款和个人贷款 
```
data6=bank[['deposit','housing','loan']]
data6['deposit']=LabelEncoder().fit_transform(data5['deposit'])
data6['housing']=LabelEncoder().fit_transform(data6['housing'])
data6['loan']=LabelEncoder().fit_transform(data6['loan'])
corrmat=data6.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corrmat,annot=True,cmap=sns.diverging_palette(220, 20, as_cmap=True))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-7cc579383d0c7381.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以得知：
- 整体来说，是否有定期存款业务和住房贷款、个人贷款的相关性都不大，或者说成负的弱相关关系（一般来说，0-0.09为没有相关性，0.3-弱，0.1-0.3为弱相关，0.3-0.5为中等相关，0.5-1.0为强相关。）；
- 定期存款业务和住房贷款的相关性系数为-0.2，可以认为拥有住房贷款，则比较不容易开设定期存款业务。
### 六、分类模型 
##### 6.1 模型概述 
###### 6.1.1 模型目标
构建一个分类模型，能够预测是否开通定期存款业务，采用的算法有：
（1）决策树；
（2）贝叶斯/费舍尔分类；
（3）神经网络；
（4）SVM

###### 6.1.2 建模过程
- 分层抽样:

![image.png](https://upload-images.jianshu.io/upload_images/18032205-5179b23e13cd3dd5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因为正负种类比例趋于1：1，所以不做处理
- 数据格式处理：
把几个字符串变量处理为字符变量：
```
bank=pd.read_csv('/Users/ranmo//Desktop/数据分析案例/银行营销/bank.csv')
bank_spss=bank

bank_spss['job']=LabelEncoder().fit_transform(bank_spss['job'])
bank_spss['marital']=LabelEncoder().fit_transform(bank_spss['marital'])
bank_spss['education']=LabelEncoder().fit_transform(bank_spss['education'])
bank_spss['default']=LabelEncoder().fit_transform(bank_spss['default'])
bank_spss['housing']=LabelEncoder().fit_transform(bank_spss['housing'])
bank_spss['loan']=LabelEncoder().fit_transform(bank_spss['loan'])
bank_spss['contact']=LabelEncoder().fit_transform(bank_spss['contact'])
bank_spss['month']=LabelEncoder().fit_transform(bank_spss['month'])
bank_spss['poutcome']=LabelEncoder().fit_transform(bank_spss['poutcome'])
bank_spss['deposit']=LabelEncoder().fit_transform(bank_spss['deposit'])

bank_spss.to_csv(path_or_buf='/Users/ranmo//Desktop/数据分析案例/银行营销/bank_spss.csv')
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-f77e2c9b0d258b97.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 特征处理，特征降维、特征缩放及连续值处理：
首先是特征维度上面，只有16+1个变量，维度不算大，因此不做降维；
特征缩放我没做，理论上肯定是要反复验证并确定缩放比例的；
连续值处理直接交给SPSS了，我人工就不作处理了。。。
ps,特征缩放本质上就是理清各个特征之间的关系，比如说上一次营销活动的成功与否与是否有存款业务没有什么关系？那么在分类过程中是否也要考虑这些变量？。。。umm，我没做缩放，肯定会影响模型精度的。。。
- 建立模型，交叉验证模型精度
要输出分类结果以及置信度
交叉验证（选定70%训练集、30%测试集）：
（1）每一次验证，会确定一个模型（参数不一样），同时输出一组confusion matrix（混淆矩阵/误差矩阵）以及ROC曲线，并确定模型的最佳分类阈值；
（2）交叉验证完毕后，得到最终的模型精度，confusion matrix和roc是加权综合的么？
（3）最终是利用全部样本构建分类预测模型。
- 比较确定最佳模型
利用各个模型交叉验证后的精度确定最佳模型，并利用全部样本构建分类模型。
###6.2 决策树
- 决策树模型选的CHAID模型
[https://blog.csdn.net/sjpljr/article/details/70169159](https://blog.csdn.net/sjpljr/article/details/70169159)
[https://www.jianshu.com/p/807b2c2bfd9b](https://www.jianshu.com/p/807b2c2bfd9b)
不同于C4.5、C5.0，CHAID模型采用卡方校验来分类。
- 交叉验证
因为参数设置上没有验证的次数，和样本比例选择，所以不太确定spss的交叉验证机制，最后选择的分割样本。

![image.png](https://upload-images.jianshu.io/upload_images/18032205-164cdb26e502e5c6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 分类结果分析
（图太大就不放了）

决策树的第一层是：duration
决策树的第二层主要是：contact、poutcome、month
决策树第三层主要是：marital、housing、month、pdays、days

**这表明：duration、contact、month、poutcome、pdays、days都是与上一次营销活动以及近期联系紧密相关的参数，这一点上很好解释，即联系越频繁，表明其本身就是我们的优质客户和目标营销群体**

- 置信度分析
![image.png](https://upload-images.jianshu.io/upload_images/18032205-30ac3938d637caf4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
理论上应该是根据最底层子节点上的分类情况进行置信度判定，比如节点40上有163个正类，2个父类，但是全部被认定为正类，则置信度为98.8%。
- 模型精度分析
（1）混淆矩阵
软件给出的这个不是混淆矩阵哈

![image.png](https://upload-images.jianshu.io/upload_images/18032205-38b6cbcb9afcae38.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

实际的混淆矩阵是:

![image.png](https://upload-images.jianshu.io/upload_images/18032205-259120f4ee08ae26.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为什么用样本构建的分类模型也存在混淆矩阵的原因是因为，没有完全拟合（CHAID构建的决策树最大深度为3，所以拟合有限）
（2）ROC曲线
一般来说，决策树不存在ROC曲线，因为不存在分类阈值的选取问题。如果一定要有ROC曲线的话，那就是通过调整树的最大深度，以及熵增、基尼值改变的最小值来调整拟合程度，最终得到一条ROC曲线，但是对决策树似乎没有什么意义。

**ps:
spss中画ROC曲线。。接受的输入各个样本的类别以及阈值，然后根据不断调整阈值，来求得ROC的X和Y：
[http://www.sohu.com/a/144925905_165070](http://www.sohu.com/a/144925905_165070)**

### 6.3 贝叶斯/费舍尔分类

- 费舍尔判别
费舍尔判别和费舍尔分类还不一样。
费舍尔分类参见[https://blog.csdn.net/luanpeng825485697/article/details/78769233](https://blog.csdn.net/luanpeng825485697/article/details/78769233)
费舍尔判别参见[https://www.jianshu.com/p/2d8a6fa92bb5](https://www.jianshu.com/p/2d8a6fa92bb5)
所以SPSS自带的很多方法都是传统统计学那一套？《挖掘理论》中的贝叶斯实际是不完全贝叶斯（因为没有求取到真是的条件概率），而费舍尔求取的是真是的条件概率，这反而是传统统计学中的贝叶斯分类。
然而SPSS没有贝叶斯判别，只有费舍尔判别，基础理论是对于有N维特征向量的输入（这里是16维，还有一位是分类标志），找到X1、X2、X3。。。X16来使得整体的组内方差最小，组间方差最大。

![image.png](https://upload-images.jianshu.io/upload_images/18032205-357cd11155cc7e5b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/18032205-e128709ba089d5e1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 交叉验证
没有交叉验证的选项，他是将所有数据用来构造费舍尔判别器，最后再对所有数据进行误差校验。
当然肯定也可以自己对表数据进行分类，确定训练集和测试集，训练好了之后保存模型用测试集进行测试，但是整体来说用模型就是不够灵活。。。
- 分类结果分析

![image.png](https://upload-images.jianshu.io/upload_images/18032205-0a22c82947ebbad8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

会给出在0组和1组的概率（置信度），在原始表中也会生成新的列给出概率。

![image.png](https://upload-images.jianshu.io/upload_images/18032205-e7fef1c099ea6c3a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 模型精度分析
（1）混淆矩阵

![](https://upload-images.jianshu.io/upload_images/18032205-226a149a3798d99c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（2）ROC曲线
费舍尔应该是判定当处在组1的概率大于处在组0的概率是，将其分为组1，所以也可以理解为：阈值=（处在组1的概率）-（处在组2的概率），当阈值>0时，则为组1，否在为组1。
因为在spss中画ROC曲线只接受两个变量，一个是正确的状态变量，一个是特征量（不同场景不同考虑，这里是（处在组1的概率-组0的概率））。
所以在spss中额外创建一列（“转换→计算变量”），再画ROC曲线：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-df7a2bddf8b4540f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

将实际ROC数据拷贝出来处理，求取“尤登指数”：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-0698cc2b1517836f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/18032205-ddef35f75537662b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以知道：
- 首先尤登指数肯定是处在0附近的，本身模型就是求取了这样的一个分类器来保证分类最优；
- 实际的尤登指数是-0.058左右，并不是0。可以理解，因为模型本身目标可能是实现precision更高，但是尤登指数与recall又相关，所以两者本身概念就有差异。

### 6.4 神经网络
- 模型
神经网络有RBF模型和多层感知器，SPSS都可以选择。多层感知器就最多有三层隐藏函数可以完成任何分析，而多层感知器求解权重采用的是BP算法，所以现在一般称为BP神经网络，两者是两个不同方向上的概念，一个是层次，一个是算法，但是又互相指代。多层感知器的求解算法可以参考之前文档的分析。
RBF和BP当然各有各的优点，以后需要再进行详细研究，需要知道的是，RBF在算法计算上不是再用BP来求解连接权重了，求解的参数有3个：基函数的中心、方差以及隐含层到输出层的权值，所以从输入层到隐藏层已经没有权重概念了，而是求解基函数的参数，具体求解流程也可以之后再研究。
还有一点就是RBF径向基函数指的是任意一个满足Φ（x）=Φ(‖x‖)特性的函数Φ都叫做径向基函数。高斯核函数通常被作为RBF函数，但是实际上还有其他函数（当然下面这篇文章的各种激活函数并不全是基函数哈，有的是BP适用的函数）：

[https://baijiahao.baidu.com/s?id=1582399059360085084&wfr=spider&for=pc](https://baijiahao.baidu.com/s?id=1582399059360085084&wfr=spider&for=pc)

下面这个是BP的激活函数，S型和双曲正切是经常用的：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-831a77f0d6f1ec9b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面这个是RBF的激活函数：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-8f6161262060e4be.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

输出层是不能选择的，默认是恒等式；隐藏层是两个学名，一个是softmax。
反正就是采用软件很不灵活，不能够自行选用函数。
还有就是最后的输出变量，本来想将输出层设置为2个神经元，分别为属于两个分类的概率，总和为1。但是这里识别到分类变量是离散值（0和1），所以自动会输出三个值：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-60e712297fa42187.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

判断依据就是，预测为1的概率大于为0的概率的时候，则预测值为1。

PS，如果输出变量是连续值的话，SPSS会认为是做预测，从而只输出一个值，就是预测的值。

最后是采用BP跑了一下。

- 分类结果分析
就输出三个值：
分类结果、预测为1的概率、预测为0的概率。
- 置信度分析
![image.png](https://upload-images.jianshu.io/upload_images/18032205-3c4e9b3a73f9bf20.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 模型精度分析
（1）混淆矩阵

![image.png](https://upload-images.jianshu.io/upload_images/18032205-ff32dbeef8310474.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

转化一下：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-da929fd881a64cbc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（2）ROC曲线

![image.png](https://upload-images.jianshu.io/upload_images/18032205-a47b81425d58429c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 6.5 SVM
SPSS statistics没有SVM，只有SPSS modeler才有。。umm
用python掉包跑一下。
- 模型
```
from sklearn import svm
from sklearn import model_selection
x,y=np.split(bank_spss,(16,),axis=1)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,train_size=0.7)
clf = svm.SVC(C=0.8, kernel='rbf', gamma='auto_deprecated', decision_function_shape='ovr')
clf.fit(x_train, y_train)
print("SVM-输出训练集的准确率为：",clf.score(x_train,y_train))
print("SVM-输出测试集的准确率为：",clf.score(x_test,y_test))
```
显示：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-85a8659d2751269f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因为训练集的准确率为1，所以最开始是猜测过拟合了，导致测试集准确率这么低。。结果我调整模型的C和gamma，得出的测试集的准确率都是一样的，然后我一查看结果，发现所有的输入都预测为0，把所有变量做了归一化处理：

```
#做归一化
bank_spss_new=bank_spss
for i in bank_spss_new.columns:
    bank_spss_new[i]=bank_spss_new[i]/(bank_spss_new[i].max()-bank_spss_new[i].min())


x,y=np.split(bank_spss_new,(16,),axis=1)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,train_size=0.7)
clf = svm.SVC(C=0.8, kernel='rbf', gamma='auto_deprecated', decision_function_shape='ovr')
clf.fit(x_train, y_train)
print("SVM-输出训练集的准确率为：",clf.score(x_train,y_train))
print("SVM-输出测试集的准确率为：",clf.score(x_test,y_test))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-6da45e16273fb8ea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 分类结果分析

![image.png](https://upload-images.jianshu.io/upload_images/18032205-863e81d031c239f3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/18032205-65655e61e3369d15.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

有两个关键性的结果参数，一个是clf.decision_function显示的是到两个分类的距离，为正就分为1类，为负就分为0类，clf.predict显示的就是分类结果。（因为分割平面是wx+b=0啊）

- 置信度分析

clf.decision_function给出了距离分类平面的距离，理论上距离越远，则表示在某一类的概率越大啊，因为离分界面远的话，不容易发生误分类。
- 模型精度分析

（1）混淆矩阵
```
from sklearn.metrics import classification_report
y_pred=clf.predict(x_test)
print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-75c03bf9a1083491.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（2）ROC曲线
SVM画ROC可能没有什么意义，如果要画的话，那就是通过调整C和gamma的值来取一个适合的模型精度。。
### 6.6 确定最佳模型
从模型的整体预测精度来说
决策树：0.807（测试集）
费舍尔判别：0.793（全部集）
BP神经网络：0.700（测试集）
svm：0.777（测试集）
 决策树胜出！

### 七、营销建议
##### 7.1 营销目标客户群体
- 年龄：从营销结果来看，20岁以下或60岁以上的群体似乎更容易营销成功，因此可以考虑将他们作为重点的营销对象。
- 职业：学生及退休人员更加可能有定期存款，同时在营销方面取得成功，蓝领、企业家、服务者、技术员不容易推销成功，应当尽量避免向这一类人进行推销。
- 住房贷款：定期存款业务和住房贷款成负若相关性，可以认为拥有住房贷款，则比较不容易开设定期存款业务。同时，开设有住房贷款的人群整体盈余情况会比没有住房贷款的人更差。所以在下次营销中心可以面向盈余情况良好且没有住房贷款的人群。
- 通话时间：通话时间越长，客户的营销成功率明显增加，因此可以将通话时间高于平均值的客户设为目标群体。
##### 7.2 营销策略
- 营销月份：营销活动主要集中在8月、5月、2月，同时5月、2月的营销都很成功，具体原因尚不知晓，但是下一次营销活动可以参考这几个月的营销经验。
- 通话时间：通话时间的长段与用户业务率呈正相关关系，因此可以考虑通过在通话期间为潜在客户提供有趣的问卷等方式，来增加通话时间，并最终提升营销活动效率。
