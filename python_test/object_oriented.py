# coding = utf-8

class Player(object):
    def __init__(self, name, points):
        self.name = name
        self.points = points
    
    def print_points(self):
        #封装，即类在定义时指定访问内部数据的方式，这种方式通过类中定义的函数来实现
        #封装数据的函数是和类本身关联在一起的，称之为类的方法
        print('{}: {}'.format(self.name, self.points))

    def get_points(self):
        #封装后，在类中添加新的方法
        if self.points >= 80:
            print("Excellent!")
        elif self.points >= 70:
            print("good!")
        else:
            print("Come on!")
    def do(self):
        print("Player is shooting...")

class AllStar(Player):
    #继承，Player的子类，具有父类Player的全部方法
    def do(self):
        #
        print("Covered...")

def run(actor):
    actor.print_points()
    actor.print_points()

Kobe = Player('Kobe Bryant', 90)
print(Kobe)
print(Kobe.points)
print(Kobe.print_points())
print("-"*35)
print(Kobe.get_points())

Curry = AllStar('Kobe Bryant', 90)
Curry.do()
Kobe.do()
print('-'*35)

print(isinstance(Curry, AllStar))
print(isinstance(Curry, Player))  # 继承
print("-"*35)

#多态，新增任何一个父类的子类，不必对run()做任何修改，任何依赖Player作为参数的函数或者方法都可以不加任何修改即可正常运行，此为多态
run(Player("Joe", 70))
run(Curry)
run(AllStar("John", 100))




## MVC

class Model():
    services = {
        'email':{'number':1000, 'price':2},
        'sms':{'number':1000, 'price':10},
        'voice':{'number':1000, 'price':15}
    }

class View():
    def list_services(self, services):
        for svc in services:
            print(svc, '')
    def list_price(self,services):
        for svc in services:
            print('For', Model.services[svc]['number'],
                         svc, "Message you pay $",
                         Model.services[svc]['price']
            )

class Controller():
    def __init__(self):
        self.model = Model()
        self.view = View()
    def get_services(self):
        services = self.model.services.keys()
        return self.view.list_services(services)

    def get_pricing(self):
        services = self.model.services.keys()
        return self.view.list_price(services)


class Client():
    controller = Controller()
    print("service provided:")
    controller.get_pricing()
    print("service for Services:")
    controller.get_services()

if __name__ == "__main__":
    client = Client()

    

