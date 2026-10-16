import numpy as np

class Variable:
    def __init__(self,data):
        self.data = data

class Function:
    def __call__(self,input):
        x = input.data
        #这里在创建forward的时候只是占位,执行时候找这个self的forward
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self,in_data):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self,x):
        return x**2
    
class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
#注意这里使用了call不是forward,在call里面取出了data并调用了forward,并及时完成call
    #所以注意输入输出都是Variable类,调用的都是Function或者其继承子类的__call__
a = A(x)
b = B(a)
y = C(b)

print(y.data)