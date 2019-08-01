import numpy as np
def read_data(path):
    a = np.zeros(14)
    f = open(path, "r+")
    for line in f:
        line = line.split()
        for i in range(1, len(line)):
            if line[i] == '1':
                a[i - 1] = a[i - 1] + 1
    return a

def read_data2(path):
    a = 0
    f = open(path, "r+")
    for line in f:
        flag = 1
        line = line.split()
        for i in range(1, len(line)):
            if line[i] == '1':
                flag = 0
        if flag == 1:
            a = a + 1
    return a
print (read_data2("test_1.txt"))