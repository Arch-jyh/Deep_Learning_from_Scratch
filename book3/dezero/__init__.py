#dezero有一个自己的模块命名空间,可以执行代码,但是区别于实际代码
#这个空间可以导入,实际导入的是这个模块命名空间的命名
#在这个包里,有一个自己的__init__命名空间,每一个文件夹(包)都有自己的__init__
    #库是人起的名字,其实是树状文件夹,每个文件夹夹可以有自己的__init__
    #在实际.py文件里可以from或者import到那个文件夹,会自动执行那个文件夹的__init__
        #这个命名空间的实例就可以导入了
    #省去了通过绝对路径导入的繁琐

#前面用True,后面用False
is_simple_core = True

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable

else:
    from dezero.core import Variable
    from dezero.core import Parameter
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import test_mode
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    from dezero.core import Config
    from dezero.layers import Layer
    from dezero.models import Model
    from dezero.datasets import Dataset
    from dezero.dataloaders import DataLoader
    from dezero.dataloaders import SeqDataLoader

    import dezero.datasets
    import dezero.dataloaders
    import dezero.optimizers
    import dezero.functions
    import dezero.functions_conv
    import dezero.layers
    import dezero.utils
    import dezero.cuda
    import dezero.transforms

setup_variable()