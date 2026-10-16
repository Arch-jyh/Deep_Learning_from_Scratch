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
    def __call__(self,inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs
    
    def forward(self,xs):
        raise NotImplementedError()
    
    def backward(self,gys): #g是梯度的意思
        raise NotImplementedError()
    
class Add(Function):
    def forward(self,xs):
        x0,x1 = xs
        y = x0 + x1
        #有了','才是元组,所以可以直接返回y, 不带()也行 'y,'等价于元组
            #元祖的本质是看有没有','间隔
            #不能写(y)这里的()只是分组符号
                #[]一定是列表,但是()有','才是元组
            #元组的返回更稳定,适合传递(列表可能会不小心被改)
        return (y,)
    
xs = [Variable(np.array(2)),Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)