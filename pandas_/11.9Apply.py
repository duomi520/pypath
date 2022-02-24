import pandas as pd
# 用来计算日期差的包
import datetime  

path ="D:\\PyPath\\pandas_\\"
def dataInterval(data1, data2):
    d1 = datetime.datetime.strptime(data1, '%Y-%m-%d')
    d2 = datetime.datetime.strptime(data2, '%Y-%m-%d')
    delta = d1 - d2
    return delta.days

# 用来计算日期间隔天数的调用的函数
def getInterval(arrLike):  
    PublishedTime = arrLike['PublishedTime']
    ReceivedTime = arrLike['ReceivedTime']
    # 注意去掉两端空白
    days = dataInterval(PublishedTime.strip(),
                        ReceivedTime.strip())  
    return days

# 用来计算日期间隔天数的调用的函数
def getInterval_new(arrLike, before, after):  
    before = arrLike[before]
    after = arrLike[after]
     # 注意去掉两端空白
    days = dataInterval(after.strip(), before.strip()) 
    return days


if __name__ == '__main__':
    fileName = path+"NS_new.xlsx"
    df = pd.read_excel(fileName)
    df['TimeInterval'] = df.apply(getInterval, axis=1)
    df['TimeInterval'] = df.apply(getInterval_new,
                                  axis=1, args=('ReceivedTime', 'PublishedTime'))  # 调用方式一
    # 下面的调用方式等价于上面的调用方式
    df['TimeInterval'] = df.apply(getInterval_new,
                                  axis=1, **{'before': 'ReceivedTime', 'after': 'PublishedTime'})  # 调用方式二
    # 下面的调用方式等价于上面的调用方式
    df['TimeInterval'] = df.apply(getInterval_new,
                                  axis=1, before='ReceivedTime', after='PublishedTime')  # 调用方式三
    print(df.head(10))                              


# https://blog.csdn.net/qq_19528953/article/details/79348929