import numpy as np
import weakref
import contextlib


#这个没用__init__创建新的实例,代码执行时候会自动创建类对象
#Config.enable_backprop调取这个变量
#   记得这是从这个类里面调取的,不是在用类创建的实例里面调取的
#python一切都是对象,这个类也是,实例也是一个对象,但是是有归属关系的,比如属于哪个类
class Config:
    enable_backprop = True
@contextlib.contextmanager
def using_config(name,value):
    #提取属性,但是用句号的提取属性是写死的,不能改名,这里的old_value可以解析出来
    #如果没有这个属性名会报错.setattr(Config,name,old_value,默认值),若没有属性返回默认值
    #attribute是属性的意思
    old_value = getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    #不是except,不论如何都会执行
    finally:
        #这是配置这个属性,with结束还原了
        setattr(Config,name,old_value)

def no_grad():
    return using_config('enable_backprop',False)


class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self,func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self,retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs = (gxs,)  
            
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    #注意这里y是弱引用
                        #弱引用的子属性.grad是强引用,因为绑定的关系没用weakref
                    #母对象一旦销毁，挂在它身上的绑定关系（属性）会连同母体一同彻底消失。
                        #这里的的.grad是一个绑定关系只是注意y弱引用只能用()提取对象

                    y().grad = None



def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
     
    def __call__(self,*inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys,tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self,xs):
        raise NotImplementedError()
    
    def backward(self,gys):
        raise NotImplementedError()
    

 
class Square(Function):
    def forward(self,x):
        y = x**2
        return y
    
    def backward(self,gy):  
         
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
def square(x):
    f = Square()
    return f(x)

    
class Add(Function):
    def forward(self,x0,x1):
        y = x0 + x1
        return y
    
    def backward(self,gy):
        return gy,gy
    

def add(x0,x1):
    return Add()(x0,x1)


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0,x1)
y = add(x0,t)
y.backward()
print(y.grad,t.grad)
print(x0.grad,x1.grad)

with using_config('enable_backprop',False):
    x = Variable(np.array(2.0))
    y = square(x)

#python里面,遇到()会被先执行,然后才是外面的关键字
with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)