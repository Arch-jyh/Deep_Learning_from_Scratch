if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable,Parameter
'''
    x = Variable(np.array(1.0))
    p = Parameter(np.array(1.0))

    #*调用的是x的,向前计算直接调取了双方的.data
    y = x * p

    #子类必定有父类的身份能力,子类向上兼容
    print(isinstance(p, Variable))
    #相当于祖谱关系,Variable上没有Parameter,所以是False
    print(isinstance(x, Parameter))
    #Function把所有的输出都用Variable包裹起来,所以输出类型不是Parameter
    print(isinstance(p, Parameter))
    print(isinstance(x, Parameter))
    print(isinstance(y, Parameter))
'''


import numpy as np
import dezero.functions as F
import dezero.layers as L


np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y,y_pred)
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1,l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)