import pandas as pd
import numpy as np
'''
df = pd.read_csv('D:/PyPath/pandas_/FE105-12-CYS1.csv', header=0, encoding =' gb2312')
df.to_csv('D:/PyPath/pandas_/123m.csv')
'''
print("--"*16)
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20170102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

print(df2)
print(len(df2))
dates = pd.date_range('20170101', periods=7)
print(dates)
print("--"*16)
df = pd.DataFrame(np.random.randn(7, 4), index=dates, columns=list('ABCD'))
print(df)
print("========= 头部3个 ========")
print(df.head(3))
print("========= 尾部3个 ========")
print(df.tail(3))
print("========= 快速统计摘要 ========")
print(df.describe())
print("========= 转置 ========")
print(df.T)
print("========= 选择一列 ========")
print(df['A'])
print("========= 选择切片行 ========")
print(df[0:3])
#print("========= 指定选择日期 ========")
# print(df['20170102':'20170103'])
print("========= 通过轴排序 ========")
print(df.sort_index(axis=1, ascending=False))
print("========= 按值排序 ========")
print(df.sort_values(by='B'))
print("========= 按标签选择 ========")
print(df.loc[dates[0]])
print("========= 标签选择多轴 ========")
print(df.loc[:, ['A', 'B']])
# print(df.loc['20170102':'20170104',['A','B']])
print("========= 通过位置选择 ========")
print(df.iloc[3])
print(df.iloc[3:5, 0:2])
print("========= 单列的值来选择数据 ========")
print(df[df.A > 0])
print("============= copy =============== ")
df3 = df.copy()
df3['E'] = ['one', 'one', 'two', 'three', 'four', 'three', 'nine']
print(df3)
print("============= start to filter =============== ")
print(df3[df3['E'].isin(['two', 'four'])])

# https://www.yiibai.com/pandas/python_pandas_series.html
# https://www.jianshu.com/p/4345878fb316
