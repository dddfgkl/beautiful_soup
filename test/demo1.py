import numpy as np

def demo1():
    a = np.array([[0,1], [1,0]])
    a = a.max(1, keepdims=True)
    print(a)

if __name__ == '__main__':
    demo1()