#globals返回当前全局变量名的字典
    #这个__file__只有作为.py执行才出现,jupyter这种单元格运行的没有
#这一步是为了在.py情况下执行时候添加上层目录,如果是jupyter这种防止__file__没有生成出现的错误报错
#实际在代码块中,用户需要启动顶层目录才可以找到
if '__file__' in globals():
    import os,sys
    #join语法是拼接文件路径的语法,兼容全平台,结果是:.../steps/..
    #这里的.../steps/..后的..的意思是step的父目录,所以定向到了父目录
        #之前的...是省略,但是..是上一层的意思(后退一步)
        #'..'也是全平台通用
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)