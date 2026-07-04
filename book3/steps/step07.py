import numpy as np

class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self,func):
        self.creator = func

    def backward(self):
        f = self.creator
        #如果是None,自动传播结束
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward() #递归,调用自己前面的变量的backward
        

class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) #锁定了创建变量的函数实例
        self.input = input #保存输入变量
        #这个函数保存输出变量
        # 注意是一个函数不会重复利用,每次这个运算都会创建新的实例
        self.output = output 
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

#assert后面的为True正常执行,为False,assert抛出报错
#用于判断条件是否满足
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x


y.grad = np.array(1.0)
C = y.creator
b = C.input
b.grad = C.backward(y.grad)

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)

print(x.grad)

y.grad = np.array(1.0)
y.backward()
print(x.grad)