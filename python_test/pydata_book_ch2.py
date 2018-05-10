"""
# from pydatabook
# 来自bit.ly的usa.gov数据
path = "E:/MyStudy/Python/pydata_book_master/ch02/usagov_bitly_data2012-03-16-1331923249.txt"
data = open(path).readline()
print(data)

# 将json字符串转化成Python字典对象
import json
records = [json.loads(line) for line in open(path)]
print(records[0])
# print("\n")

# 访问对应的记录
print(records[0]['tz'])
# print("-"*100)

# 1.1.用纯Python代码进行时区计数
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
print(time_zones[:10])

# 遍历
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

#python标准库
from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts

counts = get_counts(time_zones)
print("方法一", counts['America/New_York'])

counts2 = get_counts(time_zones)
print("方法二", counts2['America/New_York'], "\ntime_zones的长度为", len(time_zones))


# 计算得到前十位的时区及其计数值
def top_counts(count_dict, n=10):
    value_key_paris = [(count, tz) for tz, count in count_dict.items()]
    value_key_paris.sort()
    return value_key_paris[-n:]
print(top_counts(counts))

# 利用python标准库
from collections import Counter
counts = Counter(time_zones)
print(counts.most_common(10))

# 1.2.用Pandas对时区进行计数
from pandas import DataFrame,Series
import pandas as pd
import numpy as np
frame = DataFrame(records)
print(frame)

print(frame['tz'][:10])
print("+-+-"*50)

tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])
print('=='*100)

# 把缺失值用Unknow代替
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])

#利用counts的plot方法得到一张水平条形图
tz_counts[:10].plot(kind='barh', rot=0)

#分离得到用户行为摘要
result = Series([x.split()[0] for x in frame.a.dropna()])
print(result[:5])
print(result.value_counts()[:8])

#移除agent缺失的数据
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'),
                            'Windows', 'not Windows')
print(operating_system[:5])

#对数据进行分组
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts[:10])

indexer = agg_counts.sum(1).argsort()
print(indexer[:10])

count_set = agg_counts.take(indexer)[-10:]
print(count_set)

normed_subset = count_set.div(count_set.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)


# MovieLens 1M数据集
import pandas as pd
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
# 用户信息表
users = pd.read_table('E:/MyStudy/Python/pydata_book_master/ch02/movielens/users.dat',
                      sep='::', header=None, names=unames, engine='python')
print(users[:10])

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
# 评分表
rating = pd.read_table('E:/MyStudy/Python/pydata_book_master/ch02/movielens/ratings.dat',
                       sep='::', header=None, names=rnames, engine='python')
print(rating[:5])

mnames = ['movie_id', 'title', 'genres']
# 电影信息
movies = pd.read_table('E:/MyStudy/Python/pydata_book_master/ch02/movielens/movies.dat',
                       sep='::', header=None, names=mnames, engine='python')
print(movies[:5])

# pandas的merge函数将rating跟user合并，然后将movies合并
data = pd.merge(pd.merge(rating, users), movies)
print('data的前十行是\n', data[:10], '\n')
print('data.ix的值是\n', data.ix[0])

# mean_rating = data.pivot_table('rating', rows='title', cols='gender', aggfunc='mean')  # 有问题，改为如下：
mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
print('\nmean_ratings的前5行是：', mean_ratings[:5])
# pd.pivot_table('rating', rows='title', cols='gender', aggfunc='mean')  # 求每部电影的平均得分

# 对title分组，再用size()得到一个含有各电影分组大小的Series对象
rating_by_title = data.groupby('title').size()
print('\nrating_by_title的前十行是：', rating_by_title[:10])

# 评分数据大于等于250条的电影名称
active_titles = rating_by_title.index[rating_by_title >= 250]
print('\nactive_title: \n', active_titles)

# 平均评分数
mean_ratings = mean_ratings.ix[active_titles]
print('mean_ratings:\n', mean_ratings)

# 对F列进行降序排列
top_female_rating = mean_ratings.sort_values(by='F', ascending=False)
print('\ntop_female_rating:\n', top_female_rating[:10])

# 2.1计算评分分歧
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
print('\nsorted_by_diff: ', sorted_by_diff[:15])
print('\nrevised_sorted_by_diff:', sorted_by_diff[::-1][:15])
print('\nrevised_sorted_by_diff2:', sorted_by_diff[-15:])

# 根据电影名称分组的得分数据的标准差
rating_std_by_title = data.groupby('title')['rating'].std()
# 跟据active_title进行过滤
rating_std_by_title = rating_std_by_title.ix[active_titles]
print(rating_std_by_title.sort_values(ascending=False)[:10])   # 与书中有差异

"""


# 1880-2010年间全美婴儿姓名
import pandas as pd
import numpy as np
import os
from pylab import *
print(os.getcwd())
os.chdir("E:\\MyStudy\\Python\\pydata_book_master\\ch02\\names")
print(os.getcwd())
names1880 = pd.read_csv('yob1880.txt', names=['name', 'sex', 'births'])
# print(names1880)
print(names1880.groupby('sex').births.sum())

# 将所有的数据都组装到一个数据框中
years = range(1880, 2011)
pieces = []
columns = ['names', 'sex', 'births']

for year in years:
    path = 'yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)

names = pd.concat(pieces, ignore_index=True)  # concat默认是按行将多个DataFrame组合到一起，指定ignore_index = True,是不保留read.csv保留的行号
print(len(names))
print(names[:10])

# 聚合
total_births = pd.pivot_table(names, index='year', columns='sex', aggfunc=sum)
print(total_births)
total_births.plot(title='Total births by sex and year')
# show()


# 加多一个prop列
def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)
print(names[:10])
# 有效性检查，验证所有分组的prop总和是否为1
print(np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1))


# 取出该数据的子集
def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)

# print(grouped[:10])
print('top1000[:10]\n', top1000[:10])

# 分析命名趋势
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
print('boys[:10]:\n', boys[:10])

total_births = top1000.pivot_table('births', index='year', columns='names', aggfunc=sum)
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of birth per year")

# 评估命名多样性的增长
table = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex', yticks=np.linspace(0.5, 1.2, 13), xticks=range(1880, 2020, 10))
df = boys[boys.year == 2010]
print('df')
print(df[:10])

# 多少个名字的人数加起来达到50%
prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
print('prop_cumsum')
print(prop_cumsum[:10])

print(prop_cumsum.searchsorted(0.5))
df = boys[boys.year == 1900]
in1990 = df.sort_values(by='prop', ascending=False).prop.cumsum()
print(in1990.searchsorted(0.5) + 1)


#
def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().searchsorted(q) + 1

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
print(diversity.head())
#diversity.plot(title='Number of popular names in top 50%')
#show()


# 最后一个字母的变革(有问题，未解决)
'''
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
table = names.pivot_table('births', index=last_letters,
                          columns=['sex', 'year'], aggfunc=sum)
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
print(subtable.head())
'''
# 变成女孩名字的男孩名字（以及相反的情况）----(有问题，不会，暂未解决)
all_names = top1000.name.unique()
print(all_names)
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all.names[mask]
print(lesley_like)

# 过滤
filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered.groupby('name').births.sum())














