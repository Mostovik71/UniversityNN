# a, *b = 'abc' #a='a', b=['b', 'c']
# *a, b = 'abc' #a=['a','b'], b = 'c'
# a,*b=[1,2,3]
# if __name__ == '__main__':
#     print(f'{a=}')
#     print(f'{b=}')
#     # print(f'{c=}')
def example(a,b,c):
    print(a)
    print(b)
    print(c)
def my_print(*args,**kwargs):
     print(kwargs)
if __name__ == '__main__':
    # example(*[1,2,3])
    my_print(1,2,3,4,5,end='!')
    #print(1,2,3,4,5,end='!')
#**{end:'-',sep:':'} - ключ=аргумент, значение = значение аргумента