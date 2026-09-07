if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
#子程序进程,可以让python运行命令行代码
import subprocess
#处理http请求,下载文件
import urllib.request
import numpy as np
from dezero import as_variable
from dezero import Variable


#_开头表示这是内部函数,不暴露给外部
def _dot_var(v,verbose = False):
    dot_var = '{}[label = "{}",color=orange,style=filled]\n'

    name = '' if v.name is None else v.name
    #这里if后主看是 ...and... 不是只看 ...is...
    #比较运算(is)的优先级>逻辑运算(and)
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    #format把变量按照顺序填充到占位符中
    #id()是python内置函数,提取的id是这个对象的,只要对象还在id就不会变
    return dot_var.format(id(v),name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    #f.__class__.__name__提出了名字属性
    ret = dot_func.format(id(f),f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x),id(f))
    for y in f.outputs:
        #y是弱引用,所以是y()
        ret += dot_edge.format(id(f),id(y()))

    return ret


def get_dot_graph(output,verbose = True):
    txt = ''
    funcs = []
    #
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            #sort直接收一个参数的函数,利用返回值自动排序参数
            #因为绘图不需要函数的代际大小,所以注释掉了
            #funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output,verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x,verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g{\n' + txt + '}'


def plot_dot_graph(output,verbose = True,to_file = 'graph.png'):
    dot_graph = get_dot_graph(output,verbose)

    tmp_dir = 