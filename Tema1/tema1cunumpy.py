from array import array
import numpy as np
import re 


def solveSystem() : 
    file = open('tema1.txt', 'r')
    ecuations = file.readlines()
    matrixx = []
    for ecuation in ecuations : 
        coef = re.findall(r'[0-9\-\+]+', ecuation)
        coef =[int(i) for i in coef]
        matrixx.append(coef)
    A = []
    B = []
    line = []
    for j in range(3): 
        line = []
        for i in range(len(matrixx[0]) - 1) :
            line.append(matrixx[j][i])
        line = np.asarray(line)
        A.append(line)
        B.append(matrixx[j][3])
    detA = np.linalg.det(A)
    print(f'detA = {detA}')
    A = np.asarray(A)
    At = A.transpose()
    print(f'At = : {At}')
    At = np.matrix(At)
    Asteluta = At.getH()
    print(f'Asteluta = {Asteluta}')
    Ainv = np.linalg.inv(A)
    print(f'Ainv: {Ainv}')
    X = np.dot(Ainv, B)
    print(f'Sol: \n x = {X[0]} \n y = {X[1]} \n z = {X[2]}')


solveSystem()