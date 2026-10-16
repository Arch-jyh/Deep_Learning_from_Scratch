if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

#获取训练数据
x,t = dezero.datasets.get_spiral(train = True)
model = MLP((hidden_size,3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
#ceil用于小数点向上取整,3.1>>>4 不是四舍五入
max_iter = math.ceil(data_size/batch_size) 

for epoch in range(max_epoch):
    #随机重新排数据
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i+1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y,batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f'%(epoch + 1,avg_loss))


h = 0.001
x_min,x_max = x[:,0].min() - .1,x[:,0].max() + .1
y_min,y_max = x[:,1].min() - .1,x[:,1].max() + .1


xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

X = np.c_[xx.ravel(),yy.ravel()]

with dezero.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data,axis = 1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx,yy,Z)

N,CLS_NUM = 100,3
markers = ['o','x','^']
colors = ['orange','blue','green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0],x[i][1],s = 40,marker = markers[c],c = colors[c])
plt.show()