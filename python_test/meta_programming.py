#! /usr/bin/python
# -*- coding:utf-8 -*-

class Main(object):
    _dict = dict()

    def __new__(cls):
        if 'key' in Main._dict:
            print('Exists')
            return Main._dict['key']
        else:
            print('New')
            return object.__new__(cls)

    def __init__(self):
        print("Init")
        Main._dict['key'] = self


class Class_method(object):
    bar = 1
    #@classmehtod
    def class_foo(cls):
        print("Hello", self)
        print(cls.bar)
        print(class_method.bar)


# 析构
class Simple(object):
    def __init__(self):
        print("constructor called, id = {id}".format(id=id(self)))
        print("Hello {name} !".format(name="James"))

    def __del__(self):
        print("destructor called, id = {0}".format(id(self)))

    def func(self):
        print("new called, id={0}".format(id=id(self)))
        print("Simple func!")

# super()方法  http://www.runoob.com/python/python-func-super.html

class Parent():
    def __init__(self):
        self.parent = "I'm the parent;"
        print('Parent')
    def bar(self, message):
        print("{} from Parent".format(message))

class FooChild(Parent):
    def __init__(self):
        super(FooChild, self).__init__()
        #super(FooChild,self) 首先找到FooChild的父类（即FooParent类），然后把类FooChild的对象转换为类 FooParent 的对象
        print("child")
    def bar(self, message):
        super(FooChild, self).bar(message)
        print('Child bar function')
        print(self.parent)
  
# python元编程 http://python.jobbole.com/85721/
# new(cls, args, *kwargs) 创建对象时调用，返回当前对象的一个实例;注意：这里的第一个参数是cls即class本身
# init(self, args, *kwargs) 创建完对象后调用，对当前对象的实例的一些初始化，无返回值,即在调用new之后，根据返回的实例初始化；注意，这里的第一个参数是self即对象本身【注意和new的区别】
# call(self, args, *kwargs) 如果类实现了这个方法，相当于把这个类型的对象当作函数来使用，相当于 重载了括号运算符

class Test():
    def __init__(self, *args, **kwargs):
        print("init")
        # super(Test, self).__init__(*args, **kwargs)
    def __new__(cls, *args, **kwargs):
        print("new", cls)
        return super(Test, cls).__new__(cls, *args, **kwargs)
    def __call__(self, *args, **kawrgs):
        print("call")

# 内建函数的重写
class Pair():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        # return 'Pair({},{})'.format(self.x, self.y)
        return 'Pair(%s, %s)' % (self.x, self.y)
        supper()

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

# 类属性的委托访问, __getattr__
class A:
    def f_one(self, x):
        print("1")
    def f_two(self):
        return("2")

class B(A):
    def __init__(self):
        self._a = A()
    def f_three(self):
        return("f_three")
    def __getattr__(self, name):
        return getattr(self._a, name)


if __name__ == "__main__":

    # a = Simple()
    # b = Simple()

    # print(Class_method.bar)
    # Class_method.class_foo
    #main = Main()
    #main_1 = Main()
    #main_2 = Main()
     
    print("----create a new object-----")
    test = Test()
    print("----call func-----")
    test()


    foochild = FooChild()
    print('-------------------')
    foochild.bar("HelloWorld")
