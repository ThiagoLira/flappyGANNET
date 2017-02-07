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


def getDumpedNeuralNet():
    with open('nn.pkl', 'rb') as input:
        return pickle.load(input)



weights = getDumpedNeuralNet()
