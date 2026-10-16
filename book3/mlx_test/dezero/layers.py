import os
import weakref
import numpy as np
import dezero.functions as F
from dezero.core import Parameter


class Layer:
    def __init__(self):
        #集合没有顺序,所以无法索引取值
        #集合会自动去重,同内存id的继续赋值会被忽略
        self._params = set()

    #用于赋值操作:obj.xxx = yyy
        #这里的name变成了字符串'xxx' value是yyy
        #__setattr__是这个复制操作的扩展方法,没有这个方法赋值也可以进行
            #属于一个拦截功能,可以执行特殊操作
    #但如果定义了__setattr__里面没用,super().__setattr__(name,value)
        #导致赋值失败,这个__setattr__已经吧赋值拦截了,默认赋值操作移交给了super()的继承
    def __setattr__(self,name,value):
        if isinstance(value,(Parameter,Layer)):
            #记录这个命名字符串
            self._params.add(name)
        #工程里推荐前面的配置完成后再进行super,和之前的隔离
        #每个属性其实在.name的赋值里面,都存在了字典里面,除了.name提取还可以用
            #layer.__dict__[name] name字符串的方式提取value
        super().__setattr__(name,value)

    def __call__(self,*inputs):
        outputs = self.forward(*inputs)
        #输出一个值不是元祖,这里用元祖包裹
        if not isinstance(outputs,tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    #靠继承的layer类去定义forward
    def forward(self,inputs):
        raise NotImplementedError()

    #在yeild的作用下,只要有yeild,这个就是生成器
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj,Layer):
                #这里等价于for循环
                yield from obj.params()
            else:
                #yield和return类似,但可以返回多个值
                #return是结束处理返回值,如果return在yield后,便是即刻停止的意思
                #yield是暂停处理返回值
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()




class Linear(Layer):
    def __init__(self,out_size,nobias = False,dtype = np.float32,in_size = None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None,name = 'W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size,dtype = dtype),name = 'b')

    def _init_W(self):
        I,O = self.in_size,self.out_size
        W_data = np.random.randn(I,O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self,x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        y = F.linear(x,self.W,self.b)
        return y