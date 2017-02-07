#Genome is the list of weight vectors i.e. NeuralNetwork.weight object
from random import randrange,uniform,randint
from nnet import NeuralNetwork
import copy


##################################################################

#make a random mutation in ONE random weight of nnet
def mutate(genome):
    #select random weight list
    listWeights = genome[randrange(0,len(genome))]
    #select random weight from that list
    index = randrange(0,len(listWeights))
    #mutate it by mult by random float between -2 and 2
    #(Math.random() - 0.5) * 3 + (Math.random() - 0.5);
    listWeights[index] = listWeights[index]*uniform(-.89, .89)   + uniform(-1, 1)


#create a new random NeuralNetwork genome
#at present hardcoded to FlappyBird
def createParent():
    nn = NeuralNetwork([8,16,1])
    return nn.weights

#create offspring from two genomes by averaging both weights
#return new genome
def procreate(genome1,genome2):
    offspring1 = copy.deepcopy(genome1)
    offspring2 = copy.deepcopy(genome2)
    contList=-1
    for e in zip(genome1,genome2):
        contList=contList+1
        contWeight=0
        cutValue = randint(0,len(e))
        for pair in zip(*e):
            #swap weights for every even number
            if(contWeight>cutValue):
                offspring1[contList][contWeight]=pair[1]
                offspring2[contList][contWeight]=pair[0]

            contWeight=contWeight+1

    return offspring1,offspring2

#return new genome from two best fitted specimens
def procreate2(genome1,genome2):
    offspring1 = copy.deepcopy(genome1)
    offspring2 = copy.deepcopy(genome2)
    contList=-1

    layer = randint(0,len(genome1)-1)


    e = zip(genome1,genome2)[layer]

    contWeight=0
    cutValue = randint(0,len(e[1]))

    for pair in zip(*e):

        if(contWeight>cutValue):
            offspring1[layer][contWeight]=pair[1]
            offspring2[layer][contWeight]=pair[0]

        contWeight=contWeight+1

    return offspring1
