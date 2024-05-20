import random as rd
import math as m
from RBF import NeuralNetwork as RBF


#best weights 
individual = [-0.5105659020180119, 15.49147819249623, 0.005934460785414088, -6.786412852892547, -0.23674895769820442, -2.4341168236263835, -1.7659450922056692, 0.11694509025461054]
rbfNet = RBF(individual, 1, 3)


def hillClimbRestart(seed, time):
    S = rd.random()
    best = S
    for i in range(1, 200):
        for i in range(1, time):
            R = tweak(0, 1, 1, copy(S), 0, 1)
            if calcFunc(R) < calcFunc(S):
                S = R
        if calcFunc(S) < calcFunc(best):
                best = S
        S = rd.random()
    print (str(seed) + ": " + str(best))
    return best


def hillClimbRestartNeural(seed, time):
    S = rd.random()
    best = S
    for i in range(1, 200):
        for i in range(1, time):
            R = tweak(0, 1, 1, copy(S), 0, 1)
            if calcFuncTwo(R) < calcFuncTwo(S):
                S = R
        if calcFuncTwo(S) < calcFuncTwo(best):
                best = S
        S = rd.random()
    print (str(seed) + ": " + str(best))
    return best
    
def tweak(min, max, p, S, LB, UB):
    r = .005
    negR = r * -1
    for i in range(1, 200):
        rand = rd.random()
        if p > rand:
            n = 1
            while not((min <= S + n) and (S + n <= max)):
                n = LB + rd.uniform(negR, r) * (UB - LB)
            S = S + n
    return S 

def copy(S):
    R = S
    return R


def calcFunc(x):
    valueOne = (6 * x - 2) ** 2
    valueTwo = m.sin(12 * x - 4)
    return valueOne * valueTwo



def calcFuncTwo(x):
    return rbfNet.output([x])

def algorithmReal():
    for i in range(1, 3):
        rd.seed(i)
        hillClimbRestart(i, 50)
        
def algorithmRBF():
    for i in range(1, 10):
        rd.seed(i)
        hillClimbRestartNeural(i, 50)
        
print ("Normal: ")       
algorithmReal()
print("\nRBF: ")
algorithmRBF()

