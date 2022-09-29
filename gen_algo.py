import numpy as np
import matplotlib.pyplot as plt

def fitness(test, keyword):
    score = 0
    for i in range(len(test)):
        if test[i] == keyword[i]:
            score += 1
    return score

def gen_pop(N, S):
    population = []
    for i in range(N):
        individu = []
        for _ in range(S):
            individu.append(chr(np.random.randint(97, 124)))
        population.append(individu)
    return population

def crossover(i1, i2, num=1):
    for i in range(num):
        pos = np.random.randint(0, len(i1))
        temp = i1[pos]
        i1[pos] = i2[pos]
        i2[pos] = temp
    return i1

def mutation(i, num=1):
    for n in range(num):
        pos = np.random.randint(0, len(i))
        i[pos] = chr(np.random.randint(97, 124))
    return i

def choose(pop):
    dis = []
    for ind in pop:
        fit = fitness(ind)
        for f in range(fit):
            dis.append()

def sort_pop(pop):
    scored_pop = []
    for i in pop:
        fit = fitness(i, keyword)
        scored_pop.append((i, fit))
    scored_pop.sort(key=lambda x:x[1],)
    return scored_pop

if __name__=='__main__':
    
    N = 10000
    perf = []
    n_iter = 60
    keyword = "vieilabricotiermediteraneendesanremoauxfruitsdelicieux"
    S = len(keyword)
    pop = gen_pop(N, S)
    for i in range(n_iter):
        sorted_pop = sort_pop(pop)
        if sorted_pop[-1][1] == S:
            break
        pop = []
        l = 5 * len(sorted_pop)//6
        while len(pop) < l:
            r1 = np.random.randint(3*len(sorted_pop)//4, len(sorted_pop))
            r2 = np.random.randint(3*len(sorted_pop)//4, len(sorted_pop))
            ind = crossover(sorted_pop[r1][0], sorted_pop[r2][0], 2)
            # ind = mutation(ind, 2)
            pop.append(ind)
        # pop += gen_pop(N//6, S)
        
        perf.append(sorted_pop[-1][1])

        while len(pop) < N:
            r1 = np.random.randint(7*len(sorted_pop)//8, len(sorted_pop))
            ind = sorted_pop[r1][0].copy()
            ind = mutation(ind, 4)
            pop.append(ind)

    # sorted_pop = sort_pop(pop)

    print("Num iter: ", i)
    print("Keyword found is : ", sorted_pop[-1])

    plt.figure()
    plt.plot(perf)
    plt.savefig("perf.png")
