from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
import struct
from nnet import NeuralNetwork
import os
from genetic import *
import pickle
import time
import random



def process_state(state):
    return np.array([ state.values() ])


def dumpNeuralNet(nn):
    with open('nn.pkl', 'wb') as output:
        pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)

def getDumpedNeuralNet():
    with open('nn.pkl', 'rb') as input:
        return pickle.load(input)


#fit function for specimen
def oneRun(actionSet,game,genome,minValues,maxValues):
    game.reset_game()
    agent = NeuralAgent(actionSet,weights=genome)
    while(not game.game_over()):
            score=game.score()
            action = agent.pickAction(game.getGameState(),minValues,maxValues)
            teste=game.act(action)


    return game.score()

class geneticFB():
    def __init__(self,populationSize,mutationRate):
        print("bla")

class NeuralAgent():
    """
        This is our neural agent. It picks actions determined by the NeuralNetwork!
    """
    def __init__(self, actions,weights):
        self.actions = actions
        #number of state variables in FlappyBird
        self.nn = NeuralNetwork([8,16,1],weights=weights)

    #normalize inputs and feedfoward NeuralNetwork to pick action
    def pickAction(self, state,minValues,maxValues):

        stateValues = state.tolist()[0]

        for i in range(len(stateValues)):

            #update minimum value
            if(minValues[i]>stateValues[i]):
                    #print("updated min")
                    minValues[i]=stateValues[i]
            #update max value
            if(maxValues[i]<stateValues[i]):
                    #print("updated max")
                    maxValues[i]=stateValues[i]
            try:
                output = [(stateValues[i]-minValues[i])/(maxValues[i]-minValues[i]) for i in range(len(stateValues))]
            except ZeroDivisionError:
                print("Divided by zero!")
                output = [1 for i in range(len(stateValues))]

        out = self.nn.eval(output)

        if(out[0]>0.5):
            action = 0
        else:
            action = 1

        #out.index(max(out))
        return self.actions[action]


#wrapper to oneRun function so the only argument needed is the Genome
def fitFuncWrapper(genome):
    return oneRun(actionSet,p,genome,minValues,maxValues)


#inicialize population
#global variables

g = FlappyBird()
p = PLE(g, state_preprocessor=process_state)
actionSet = p.getActionSet()
popMax = 16
pop = [[createParent(),-1] for i in range(popMax)]
start_time = time.time()
mutationRate = 1000
count=0
countShow=0


#Initial values for the state variables
minValues = [1 for i in range(8)]
maxValues = [0  for i in range(8)]


#number of specimens that stay in each generation
# popMax - N is thrown away
N=15


#run for 1000 generations and stop

pop[0][0] = getDumpedNeuralNet()

for i in range(10000000):
    mutationRate = mutationRate - 1
    count= count +1
    countShow= countShow +1
    #evaluate!
    for tuple in pop:
        #evaluate just new ones
        if(tuple[1]==-1):
            tuple[1] =  fitFuncWrapper(tuple[0])

    #sort list by fitness value
    pop.sort(key=lambda tup:tup[1],reverse=True)

    print("Fit values for this generation: "),
    for i in range(0,len(pop)):
        print(pop[i][1]),
    print("")
    #show best individual
    #p.display_screen=True
    bestFit = pop[0][1]

    if(countShow>100):
        p.display_screen=True
        fitFuncWrapper(pop[0][0])
        p.display_screen=False
        countShow=0
        #dump best fit!
        dumpNeuralNet(pop[0][0])

    #get N best genomes
    #remove the worst of this generation
    pop.pop()

    firstBest = copy.deepcopy(pop[0][0])
    secondBest = copy.deepcopy(pop[1][0])


    newIndividual = procreate2(firstBest,secondBest)
    for i in range(mutationRate):
        mutate(newIndividual)

    while(len(pop)<popMax):
        pop.insert (0,[newIndividual,-1])

    elapsed_time = time.time() - start_time

    print(str(count)+ "-th Genearion Best fit: "+ str(bestFit) +" Time Elapse:"+ str(elapsed_time))
