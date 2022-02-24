import pandas as pd
import numpy as np

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
                     'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
            'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2],
            'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
            'Points': [876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]}
df = pd.DataFrame(ipl_data)
print(df)
print("==========查看分组==========")
print(df.groupby('Team').groups)
print("==========迭代遍历分组==========")
grouped = df.groupby('Year')
for name, group in grouped:
    print(name)
    print(group)
print("==========选择一个分组==========")
print(grouped.get_group(2014))
print("==========聚合函数==========")
print(grouped['Points'].agg(np.mean))
print("==========多个聚合函数==========")
grouped = df.groupby('Team')
agg = grouped['Points'].agg([np.sum, np.mean, np.std])
print (agg)
print("==========转换==========")
grouped = df.groupby('Team')
score = lambda x: (x - x.mean()) / x.std()*10
print (grouped.transform(score))
print("==========过滤==========")
filter = df.groupby('Team').filter(lambda x: len(x) >= 3)
print (filter)