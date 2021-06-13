from ComOp.tsp_helper_class import TravelingSalesmanProblem
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import array
from ComOp import elitism

TSP_NAME = "bayg29"
tsp = TravelingSalesmanProblem(TSP_NAME)


def tspDistance(ind):
    return tsp.getTotalDistance(ind),


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("randomOrder", random.sample, range(len(tsp)), len(tsp))
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
toolbox.register("evaluate", tspDistance)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(tsp))

max_generations = 500
population_size = 300
hall_of_fame_size = 30
probability_crossover = 0.9
probability_mutation = 0.1
random.seed(42)


def main():
    population = toolbox.populationCreator(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(hall_of_fame_size)
    population, logbook = algorithms.eaSimpleWithElitism(population, toolbox, cxpb=probability_crossover,
                                              mutpb=probability_mutation, ngen=max_generations,
                                              stats=stats,
                                              halloffame=hof,
                                              verbose=True)

    best = hof.items[0]

    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    plt.figure(1)
    tsp.plotData(best)

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


def main_elitism():
    population = toolbox.populationCreator(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(hall_of_fame_size)
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=probability_crossover,
                                                      mutpb=probability_mutation, ngen=max_generations,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    best = hof.items[0]

    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    plt.figure(1)
    tsp.plotData(best)

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
