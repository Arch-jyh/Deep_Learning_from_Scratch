import numpy as np

class Variable:
    def __init__(self,data):
        self.data = data

#用array创建narray类
#   int等python的自带类型也是类,这里用小写因为属于更基础的类型
#       narray类也用了这里的思想,表示自己很基础
#   直接用narray创建太基础了,用array函数更适合,需要的初始化参数更少
data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)