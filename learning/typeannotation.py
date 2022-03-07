from typing import List, Union, Optional, Any
#Any - should be any
#Optional - may be NaN
#Union - единение, для двух и более типов, for example (a: Union[int, float] - a should be int or float
def calc(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:#('-> int' means that function will return 'int' type;a:int, b:int means that a,b should be 'int' type)
    return a + b
def to_int(a_list:list[str])->list[int]: #List[int] or (list[int] (python 3.9)) - list of ints
    return [int(i) for i in a_list]

if __name__ == '__main__':
    print(calc(5, 2.37))
    print(to_int(['1', '2'])[0])
