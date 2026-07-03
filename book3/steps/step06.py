import numpy as np

class Variable:
    def __init__(self,data):
        self.data = data
        #初始化求出来的导数的值,和data一样都是narray的数据类型
        self.grad = None
        

class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input #保存输入变量
        return output
    
    def forward(self,x):
        raise NotImplementedError()
    
    def backward(self,gy):
        raise NotImplementedError()
    
    
class Square(Function):
    def forward(self,x):
        return x**2

    def backward(self,gy):
        x = self.input.data
        gx = 2 * x * gy #gy是一个narray实例(array数组)
        return gx


class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
    def backward(self,gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)


y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)