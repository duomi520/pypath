import pandas as pd
path ="D:\\PyPath\\pandas_\\"
io =path+"flush.xlsx"
data = pd.read_excel(io)
g=data.groupby('材质').groups['PP+30%GF']
data.ix[g].to_excel(path+"flush_g.xlsx")

# https://blog.csdn.net/tongxinzhazha/article/details/78796952