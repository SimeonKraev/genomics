from deap import base, creator, tools, algorithms
from ComOp.knapsack_helper_class import Knapsack01Problem
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


knapsack = Knapsack01Problem()


def knapsackValue(individual):
    return knapsack.getValue(individual),


max_generations = 50
population_size = 50
hall_of_fame_size = 1
probability_crossover = 0.9
probability_mutation = 0.1
random.seed(42)


toolbox = base.Toolbox()
toolbox.register("evaluate", knapsackValue)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(knapsack))

toolbox.register("zeroOrOne", random.randint, 0, 1)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(knapsack))
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def main():
    population = toolbox.populationCreator(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(hall_of_fame_size)
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=probability_crossover,
                                              mutpb=probability_mutation, ngen=max_generations,
                                              stats=stats,
                                              halloffame=hof,
                                              verbose=True)

    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    print("-- Knapsack Items = ")
    knapsack.printItems(best)

    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()
