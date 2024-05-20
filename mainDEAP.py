import random as r
import numpy as np
import math
import pandas as pd

#from RBF import ANN
from RBF import NeuralNetwork as RBF
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from bdb import Breakpoint
import seaborn as sns
from nltk.sem.logic import IndividualVariableExpression

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_real", r.uniform, -5, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, n = 8) 
toolbox.register("population", tools.initRepeat, list, toolbox.individual) #, ?)

#normalize input
def normalize(MIN, MAX, inputVal):
    return (float(inputVal) - MIN) / (MAX - MIN)

#normlize output
def normalizeOutput(MIN, MAX, inputVal):
    return (((15.82 - -4.949) *(float(inputVal) - MIN)) / (MAX - MIN)) + 0

def getGraph(individual):
    x = []
    point = 0.0
    for i in range(0,100):
        x.append(point)
        point += 0.01
        
    y = []
    rbfNet = RBF(individual, 1, 3)
    for i in range(0, len(x)):
        inputVal = normalize(0, 1, x[i])
        yHat = rbfNet.output([inputVal])
        outputVal = yHat
        y.append(outputVal)
        #print ("x: " + str(x[i]) + " #support: " + str(outputVal))
        
    print( "FINAL X: "+ str(x))
    print( "FINAL OUTPUT: "+ str(y))

# Fitness Evaluation:
def evalANN(individual):
    x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y = [3.027209981231713, -0.6565767743055739, -0.639727105946563, -0.01557673369234606, 0.11477697454392392, 0.9092974268256817, -0.14943780717460267, -4.605754037625252, -4.949130440918993, 5.71195033916232, 15.829731945974109]

    #print("\n\nindividual: " + str(individual))
    rbfNet = RBF(individual, 1, 3)
    fitness = 0
    yHat = 0
    for i in range(0, len(x)):
        inputVal = normalize(0, 1, x[i])
        yHat = rbfNet.output([inputVal])
        fit = (y[i] - yHat) ** 2
        fitness = fitness + (fit)
        #print(" yHat: " + str(yHat) + " x: " + str(x[i]) + " y: " + str(y[i]))
        
    print ("speed: " + str(fitness))
    return (fitness * -1,)

toolbox.register("evaluate", evalANN)



    

#custom two point crossover mutation
def twoPointCrossover(parentOne, parentTwo, alpha = 0.0):
    value = r.random()
    if value > alpha:
        breakpoint = len(parentOne)
        while breakpoint > (len(parentOne) - 2) or breakpoint % 2 != 0:
            print(breakpoint)
            breakpoint = r.randint(0, len(parentOne) - 1)
            
        for i in range(breakpoint, breakpoint + 2):
            temp = parentOne[i]
            parentOne[i] = parentTwo[i]
            parentTwo[i] = temp
        
    return (parentOne, parentTwo)  

toolbox.register("mate", twoPointCrossover, alpha = 0.3)
toolbox.register("mutate", tools.mutGaussian, mu = 0.0, sigma = 1.5, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == "__main__":
        pop = toolbox.population(n=300)
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.05, ngen= 4000)
        
t = tools.selBest(pop, k=1)[0]

print (str(t))
print (str(getGraph(t)))
