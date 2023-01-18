from array import array
import numpy as np
import re 

def detMic(a, b, A) :
    d = np.empty([2,2])
    nd = 0
    md = 0
    for i in range(3) : 
        for j in range(3):
            if i!=a and j!=b : 
                if md == 0 : 
                    d[nd][md] = A[i][j]
                    md += 1
                elif md == 1 :
                    d[nd][md] = A[i][j]
                    md = 0
                    nd += 1
    detd = d[0][0]*d[1][1] - d[0][1]*d[1][0]
    return int(detd)
    



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
        A.append(line)
        B.append(matrixx[j][3])
    detA = A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1] - A[0][2]*A[1][1]*A[2][0] -A[0][1]*A[1][0]*A[2][2] - A[0][0] *A[1][2]*A[2][1]
    if detA == 0 : 
        print("detA = 0 ")
        return 0
    print(f'detA = {detA}')
    At = np.empty([3,3])
    for i in range(len(A[0])) : 
        for j in range(len(A)) : 
            At[i][j] = A[j][i]
    print(f'At = : {At}')
    Asteluta = np.empty([3,3])
    for i in range(len(At[0])) :  
        for j in range(len(A)) :  
            Asteluta[i][j] = ((-1)**(i+j)) * detMic(i,j,At)
    print(f'Asteluta = {Asteluta}')
    Ainv = np.empty([3,3])
    for i in range(len(Asteluta)) : 
        for j in range(len(Asteluta[0])) : 
            Ainv[i][j] = Asteluta[i][j] / detA
    print(f'Ainv = {Ainv}')
    sol = []
    buff = 0 
    for i in range(len(Ainv)): 
        for j in range(len(Ainv[0])) : 
            buff += Ainv[i][j] * B[j]
        sol.append(buff)
        buff = 0
    print(f'Sol: \n x = {sol[0]} \n y = {sol[1]} \n z = {sol[2]}')




solveSystem()