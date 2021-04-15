import numpy as np
from deap import creator, base, tools
import random

class Employee:
    pass


creator.create("Developer", Employee, position="Developer", programmingLanguage=set)

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("FitnessCompound", base.Fitness, weights=(1.0, 0.2, -0.5))

creator.create("Individual", list, fitness=creator.FitnessMax)

def sumOfTwo(a, b):
    return a + b

toolbox = base.Toolbox()
toolbox.register("incrementByFive", sumOfTwo, b=5)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)

# Population initialization
randomList = tools.initRepeat(list, random.random, 30)


def zeroOrOne():
    return random.randint(0, 1)


randomList2 = tools.initRepeat(list, zeroOrOne, 30)

toolbox.register("zeroOrOne", random.randint, 0, 1)
randomList3 = tools.initRepeat(list, toolbox.zeroOrOne, 30)

# Fitness
def someFitnessCalculationFunction(individual):
    return None  # _some_calculation_of_the_fitness


toolbox.register("evaluate", someFitnessCalculationFunction)






