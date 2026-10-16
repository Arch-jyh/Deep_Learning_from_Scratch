import numpy as np

class Variable:
    def __init__(self,data):
        self.data = data

class Function:
    #特殊的方法,如果后面有变量被这个类赋值,那么这个变量直接使用()即可调用call,当函数用
    #没有写__init__就是没有初始化,单纯创建了对象
        #默认继承了python的object类,自带空__init__所以不会报错
        #创建时候Function()及用的__init__,这里不会默认用call.
        #所以没有__init__的时候,需要先创建对象,在用()调用对象的call
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self,in_data):
        #Function类forward没有实现
        # 调用Function或者后面继承Function类的新的类没支持,调用forward都会报错
        raise NotImplementedError()
    

#注意这里没把类的data变量取出来,所以这里的是直接处理变量的,传入类报错
class Square(Function):
    def forward(self,x):
        return x**2
    

x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)

y2 = f(x)
print(type(y2))
print(y2.data)