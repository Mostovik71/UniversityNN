class Cat:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def meow(self):
         print(f'{self.name} says: Meow!')

if __name__ == '__main__':
    tom = Cat(name='Tom',age=2)
    angela = Cat('Angela', 1)
tom.meow()