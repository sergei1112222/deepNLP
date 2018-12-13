import numpy as np
import matplotlib.pyplot as plt
import xlrd

path = "C:/Users/Sergey/Documents/MIPT/diploma/MStarIndicesMonthly_Trunc_ordered.xlsx"
pathOutMul = "C:/Users/Sergey/Documents/MIPT/diploma/outMul.txt"


def readFileData(path, row_start, col_start):
    rb = xlrd.open_workbook(path)
    sheet = rb.sheet_by_index(0)
    values = []
    for rownum in range(sheet.nrows):
        row = sheet.row_values(rownum)
        values.append(row)

    values = np.array(values)
    values = values[row_start:, col_start:]

    return values.astype(np.float32)



def calc_mul(data):
    data = np.array(data)
    data += 1
    mul_tabl = np.ones((data.shape[0],1))
    for i in range(data.shape[1] - 1):
        t_mul = data[:,:i+1]
        t_mul = np.prod(t_mul,axis = 1).reshape((data.shape[0],1))
        mul_tabl = np.hstack((mul_tabl,t_mul))
    return mul_tabl

def writeData(path, data):
    data = np.array(data)
    with open(path,'w') as f:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                f.write(data[i,j]," ")
            f.write('\n')

def modelYield(data, beta):
    data = np.array(data)
    beta = np.array(beta)
    '''beta = beta.reshape((beta.shape[0],1))

    Var = 0.1*(1/data.shape[1])*np.dot(PartDist,PartDist)
    e = np.random.normal(0,)
    returnVal = np.multiply(data,beta);
    return returnVal.sum(axis = 0)
    '''
    PartDist = np.dot(beta,data)
    return PartDist + np.random.normal(0,np.sqrt(0.2*1/len(PartDist)*np.dot(PartDist,PartDist)),len(PartDist))

def multyOperation(data, t, i):
    t = t - 1
    if t == -1:
        return 1
    else:
        mul = 1
        for i in range(t + 1):
            mul *= (1 + data[i,t])
        return mul
def sum_return(data, beta, t, product_matr):
    mulVect = product_matr[:,t]

    normalisator = 0
    for j in range(data.shape[0]):
        normalisator += beta[j]*mulVect[j]

    sum = 0

    for i in range(data.shape[0]):
        sum += ((beta[i]*mulVect[i]) / normalisator) * data[i,t]
    return sum


def betaInit(assets_numbers, N):
    beta = np.zeros(N)
    for i in range(N):
        if i in assets_numbers:
            beta[i] = 1/13
    return beta

def mainKriteria(beta, data,modelBeta):
    modelData = modelYield(data,modelBeta)
    products = calc_mul(data)
    sum = 0

    for t in range(data.shape[1]):
        sum += (modelData[t] - sum_return(data,beta,t,products))**2
    return sum



yield_values = readFileData(path,1,3)
yield_values /= 100

numAssets = [92,164,184,243,263,288,332,347,412,421,507,522,626]
beta = betaInit(numAssets, yield_values.shape[0])




def funcForGraph(z,b1,b2,data,modelBeta):
    arg = z*b1 + (1-z)*b2

    return mainKriteria(arg,data,modelBeta);


b1 = np.zeros(yield_values.shape[0])

b1[164]  =1.0
b2 = np.zeros(yield_values.shape[0])
b2[522] = 1.0
model_beta = 0.5*b1 + 0.5*b2


z = np.arange(0,1,0.1)

outG = []
for v in z:
    print(v)
    outG.append(funcForGraph(v,b1,b2,yield_values,model_beta))
fig = plt.figure()
plt.plot(z,outG)
plt.show()

