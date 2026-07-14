import numpy as np


class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self,func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x,y = f.input,f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    #*出现在函数定义里面是打包,出现在函数调用里面是解包
        #吧输入作为元组赋值到inputs里面
    def __call__(self,*inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        #is instance判断是不是这个类型,返回True或者False
        #针对forward一个元素只返回元素的情况
        if not isinstance(ys,tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        #if 条件通过,执行前面的,如果不通过,执行后面的else
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self,xs):
        raise NotImplementedError()
    
    def backward(self,gys): #g是梯度的意思
        raise NotImplementedError()
    
class Add(Function):
    def forward(self,x0,x1):
        y = x0 + x1
        return y
    

def add(x0,x1):
    return Add()(x0,x1)

a = Variable(np.array(2));b = Variable(np.array(3))
y = add(a,b)
print(y.data)