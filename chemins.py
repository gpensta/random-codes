import numpy as np
import matplotlib.pyplot as plt


def fitness(wp, path):
    score = 0
    for i in range(len(path)-1):
        score -= ((wp[0, path[i]] - wp[0, path[i+1]])**2 + (wp[1, path[i]] - wp[1, path[i+1]])**2)**0.5 
    return score

def gen_pop(N, n_wp):
    population = []
    for i in range(N):
        population.append(np.random.permutation(n_wp))
    return population

def crossover(i1, i2, num=1):
    new_i = {}
    while len(new_i) != i1.shape[0]:
        for i in range(num):
            pos = np.random.randint(0, i1.shape[0])
            temp = i1[pos]
            i1[pos] = i2[pos]
            i2[pos] = temp
        new_i = set(i1)
    return i1

def switch(i1,  num=1):
    for i in range(num):
        pos1 = np.random.randint(0, i1.shape[0])
        pos2 = np.random.randint(0, i1.shape[0])
        temp = i1[pos1]
        i1[pos1] = i1[pos2]
        i1[pos2] = temp        
    return i1

def sort_pop(pop, wp):
    scored_pop = []
    for i in pop:
        fit = fitness(wp, i)
        scored_pop.append((i, fit))
    scored_pop.sort(key=lambda x:x[1])
    return scored_pop

def display(wp, path):
    fig, ax = plt.subplots()
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    ax.scatter(wp[0, :], wp[1, :])
    for i in range(len(path) - 1):
        plt.plot([wp[0, path[i]], wp[0, path[i+1]]], [wp[1, path[i]], wp[1, path[i+1]]])
    plt.savefig("wp.png")

if __name__ == '__main__':
    n_wp = 25
    population_size = 150
    n_iter = 500

    perf = []
    


    pop = gen_pop(population_size, n_wp)
    wp = np.random.randint(0,100, (2, n_wp))
    
    for i in range(n_iter):
        sorted_pop = sort_pop(pop, wp)
        perf.append(sorted_pop[-1][1])
        print("perf : ", sorted_pop[-1][1])
        pop = []

        elite_size = 5 * len(sorted_pop)//6

        while len(pop) < elite_size:
            r1 = np.random.randint(3*len(sorted_pop)//4, len(sorted_pop))
            r2 = np.random.randint(3*len(sorted_pop)//4, len(sorted_pop))
            ind = crossover(sorted_pop[r1][0], sorted_pop[r2][0], 4)
            pop.append(ind)

        while len(pop) < population_size:
            r1 = np.random.randint(7*len(sorted_pop)//8, len(sorted_pop))
            ind = np.copy(sorted_pop[r1][0])
            ind =switch(ind, 5)
            # ind = mutation(ind, 4)
            pop.append(ind)

    print("Num iter: ", i)
    print("Keyword found is : ", sorted_pop[-1])
    #print(f"Number of operations : {i*population_size:1e}")
    #print(f"Naive requires : {26**keyword_size:1e}")

    plt.figure()
    plt.title("Performance")
    plt.xlabel("n iterations")
    plt.ylabel("score")
    plt.plot(perf)
    plt.savefig("perf.png")



    
    display(wp, sorted_pop[-1][0])