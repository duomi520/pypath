import pandas as pd
left = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
    'subject_id': ['sub1', 'sub2', 'sub4', 'sub6', 'sub5']})
right = pd.DataFrame(
    {'id': [1, 2, 3, 4, 5],
     'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
     'subject_id': ['sub2', 'sub4', 'sub3', 'sub6', 'sub5']})
print(left)
print("========================================")
print(right)
print("==========在一个键上合并两个数据帧==========")
rs = pd.merge(left, right, on='id')
print(rs)
print("==========合并多个键上的两个数据框==========")
rs = pd.merge(left, right, on=['id', 'subject_id'])
print(rs)
print("==========Left Join示例==========")
rs = pd.merge(left, right, on='subject_id', how='left')
print(rs)
print("==========Right Join示例==========")
rs = pd.merge(left, right, on='subject_id', how='right')
print(rs)
print("==========Outer Join示例==========")
rs = pd.merge(left, right, on='subject_id', how='outer')
print(rs)
print("==========Inner Join示例==========")
rs = pd.merge(left, right, on='subject_id', how='inner')
print(rs)
