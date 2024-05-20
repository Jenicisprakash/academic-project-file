
import numpy as np

class Neuron: 
    
    #constructor
    def __init__(self, funcValues):
        self.inputs = 0
        self.funcValues = list(funcValues)
    
    #adds input to neuron
    def addInput(self, value):
        self.inputs = self.inputs + value
    
    #Gaussian RBF 
    def radialBasisFunction(self, x, r, c):
        return np.exp(-1 * (((x - c) ** 2) / r**2))
    
    #sigmoid activation function
    def sigmoid(self, x, deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    #return sigmoidOutput
    def outputSigmoid(self):
        returnValue = self.sigmoid(self.inputs)
        self.inputs = 0
        return returnValue
    
    #return RBF output
    def outputRBF(self):
        x = self.inputs
        r = self.funcValues[0]
        c = self.funcValues[2]
        returnValue = self.radialBasisFunction(x, r, c)
        self.inputs = 0
        
        return self.funcValues[1] * returnValue
    
    #returns non activated output
    def nonOutput(self):
        if sum(self.funcValues) != 0:
            
            returnValue = self.inputs * self.funcValues[1]
        else:
            returnValue = self.inputs
        if self.inputs == -1:
            self.inputs = -1
        else:
            self.inputs = 0
        return returnValue
        

class NeuronLayer:
    
    #constructor
    def __init__(self, funcValues, numNeurons, includeBias):
        self.funcValues = list(funcValues)
        self.numNeurons = numNeurons
        self.neurons = []
        self.createLayer(includeBias)
    
    #create neuron layer
    def createLayer(self, includeBias = True):
        #includes a bias neuron if includeBias = True
        if (includeBias):
            if (sum(self.funcValues) != 0):
                bias = Neuron([0, self.funcValues[0]])
            else:
                bias = Neuron([])
            bias.addInput(-1)
            self.neurons.append(bias)
        
        if len(self.funcValues) >= 2:
            for i in range(1, self.numNeurons + 1):
                neuronValues = []
                neuronValues.append(self.funcValues[len(self.funcValues) - 1])
                neuronValues.append(self.funcValues[i])
                neuronValues.append(self.funcValues[i + 1])
                self.neurons.append(Neuron(neuronValues))
        else:
            for i in range(0, self.numNeurons):
                self.neurons.append(Neuron([]))

                
class NeuralNetwork:
    
    def __init__(self, funcValues, numInputs, numHiddenNeurons):
        self.funcValues = list(funcValues)
        self.numInputs = numInputs
        self.numHiddenNeurons = numHiddenNeurons
        self.layers = []
        self.createLayers()
    
    def createLayers(self):
        self.layers.append(NeuronLayer([], self.numInputs, True))
        self.layers.append(NeuronLayer(self.funcValues, self.numHiddenNeurons, True))
        self.layers.append(NeuronLayer([], 1, False))
    
    #neural network output  
    def output(self, inputs = []):
        inputLayer = self.layers[0]
        hiddenLayer = self.layers[1]
        outputLayer = self.layers[2]
        
        for i in range(0, len(inputs)):
            inputLayer.neurons[i + 1].addInput(inputs[i])
        
        #input layer to hidden layer
        for i in range(0, len(inputLayer.neurons)):
            value = inputLayer.neurons[i].nonOutput()
            for j in range(1, len(hiddenLayer.neurons)):
                hiddenLayer.neurons[j].addInput(value)
                
        #adding bias output to output layer
        hiddenBias = hiddenLayer.neurons[0]
        outputNeuron = outputLayer.neurons[0]
        toAdd = hiddenBias.nonOutput()
        
        outputNeuron.addInput(toAdd)
        
        #rest of hidden output to output layer
        for i in range(1, len(hiddenLayer.neurons)):
            value = hiddenLayer.neurons[i].outputRBF()
            
            outputNeuron.addInput(value)
        
        #return output of output layer  
        returnValue =  outputNeuron.nonOutput()
       
        return returnValue
 


    
            
        