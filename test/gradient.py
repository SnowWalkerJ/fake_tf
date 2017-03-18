from faketf import Variable


if __name__ == '__main__':
    a = Variable([1.0, 2.0, 3.0])
    b = a * 2
    c = a * b
    gradient = c.auto_derivate({a})
    print(gradient[a].eval())
