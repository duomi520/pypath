import numpy as np
import pandas as pd
#创建df对象
dict_data = {
    'key1':['a','b','c','d','a','b','c','d'],
    'key2':['one','two','three','one','two','three','one','two'],
    'data1':np.random.randint(1,10,8),
    'data2':np.random.randint(1,10,8)
}
df = pd.DataFrame(dict_data)
print(df)
print("==========单个列上应用聚合==========")
print(df['data1'].aggregate(np.sum))
print("==========多列上应用聚合==========")
print(df[['data1','data2']].aggregate([np.sum,np.mean]))
print("==========将不同的函数应用于DataFrame的不同列==========")
print(df.aggregate({'data1' : np.sum,'data2' : np.mean}))

#https://blog.csdn.net/baoshuowl/article/details/79870706

