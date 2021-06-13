from ComOp.vrp_helper_class import VehicleRoutingProblem
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ComOp.elitism import eaSimpleWithElitism
import array


TSP_NAME = "bayg29"
NUM_OF_VEHICLES = 3
DEPOT_LOCATION = 12


vrp = VehicleRoutingProblem(TSP_NAME, NUM_OF_VEHICLES, DEPOT_LOCATION)


def vrpDistance(ind):
    return vrp.getMaxDistance(ind),


toolbox = base.Toolbox()
toolbox.register("evaluate", vrpDistance)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=2.0/len(vrp))
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(vrp))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)
toolbox.register("randomOrder", random.sample, range(len(vrp)), len(vrp))
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population_size = 500
probability_crossover = 0.9
probability_mutation = 0.1
max_generations = 400
hall_of_fame_size = 30


def main_elitism():
    population = toolbox.populationCreator(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(hall_of_fame_size)
    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=probability_crossover,
                                                      mutpb=probability_mutation, ngen=max_generations,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    best = hof.items[0]

    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    print("-- Route Breakdown = ", vrp.getRoutes(best))
    print("-- total distance = ", vrp.getTotalDistance(best))
    print("-- max distance = ", vrp.getMaxDistance(best))

    plt.figure(1)
    vrp.plotData(best)

    # plot statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # show both plots:
    plt.show()


main_elitism()

