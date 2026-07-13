import numpy as np

class Variable:
    def __init__(self,data):
        if data is not None:
            #isinstance判断data对象是不是后面的类型
            #numpy计算输出如果是1D,那么可能是numpy的float的64或者32,而不是ndarray了
            if not isinstance(data,np.ndarray):
                #不是None也不是ndarray就会报错
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self,func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            #如果为None创建形状相同的梯度1,如果data是标量,grad也将会是标量
            #用这个也会自动匹配数据类型,包括浮点数的位数 按照数据类型填充1
                #容器是一个类型,内部数据还是一个类型
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x,y = f.input,f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    #判断是不是标量
        #标量是单个数值,不是数组
        #这里将其变为数组
    if np.isscalar(x):
        return np.array(x)
    return x
        

class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        #这里只有forward检查
            #为了代码一致性和后续扩展方便
            #grad进行了设计取舍,不需要as_array,因为forward的data用的多
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
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
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
    def backward(self,gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    

def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

x = Variable(np.array(1.0))
x = Variable(None)