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

    #expanduser是展开用户的意思
    #expanduser和~搭配,返回用户目录的字符串 User\Arch. (主目录)
    #用join拼接起来,这是一个跨平台工具,可以识别添加不同系统的文件夹分隔符
        #创建了.dezero的文件夹,存放缓存
    #join只负责拼接,所以可能没有文件夹
    tmp_dir = os.path.join(os.path.expanduser('~'),'.dezero')
    #exist判断是否存在
    if not os.path.exists(tmp_dir):
        #mkdir 是 make directory 创建文件夹的意思
        #识别的路径名,可以创建文件夹,文件夹的名字可以带.如aaa.aaa,识别的还是文件夹
        #.aaa是特殊的隐藏文件夹
        os.mkdir(tmp_dir)
    #一样是拼接,只是得到的字符串,没有创建文件
    graph_path = os.path.join(tmp_dir,'tmp_graph.dot')

    #open这读取了字符串已经在对应位置创建了文件了
    #open和mkdir都只创建目录最后的文件和文件夹,如果前面的路径没有会报错
    with open(graph_path,'w') as f:
        f.write(dot_graph)


    #splitext将字符串拆分成一个名字和字符串后缀 返回一个元组
        #[1]取出来的是第二项'.png' [1:]继续取出'png'
    extension = os.path.splitext(to_file)[1][1:]
    #graph_path是读取的dot文件 extension是写入的二进制的格式 to_file是输出文件名
    cmd = 'dot {} -T {} -o {}'.format(graph_path,extension,to_file)
    subprocess.run(cmd,shell = True)

    try:
        from IPython import display
        #在jupyter命令行显示图像
        return display.Image(filename = to_file)

    except:
        pass