import pandas as pd
import numpy as np
#不要尝试在迭代时修改任何对象。迭代是用于读取
N = 5
df = pd.DataFrame({
    'A': pd.date_range(start='2016-01-01', periods=N, freq='D'),
    'x': np.linspace(0, stop=N-1, num=N),
    'y': np.random.rand(N),
    'C': np.random.choice(['Low', 'Medium', 'High'], N).tolist(),
    'D': np.random.normal(100, 10, size=(N)).tolist()
})
print(df)
print("==========迭代列名==========")
for col in df:
    print(col)

print("========== iteritems()示例==========")
for key, value in df.iteritems():
    print(key,",", value)
    
print("========== iterrows()示例==========")
for row_index, row in df.iterrows():
    print(row_index, ",",row)

print("========== itertuples()示例==========")
for row in df.itertuples():
    print (row)