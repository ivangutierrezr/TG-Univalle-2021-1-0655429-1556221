import numpy as np
import networkx as nx
import math
import time
import csv

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

#Variables iniciales
start = 0 # Variables para medición de tiempo
end = 0 # Variables para medición de tiempo
currentGen = 0 # Generación actual
pop = 100 # Tamaño de población inicial
mu, sigma = 0, 0.4 # Media y desviación estándar
maxGen = 30 # Máximo de generaciones
chosenSize = 5 # Tamaño de la población a escoger para reproducción
keyCount = 100 # Variable para controlar el último valor de llave creado. Esta variable es para generar la red en NetworkX y el valor de entrada es igual al de la variable pop
maxIter = 100 # Número máximo de iteraciones para cada ciclo generacional. Se usa para correr el algorítmo múltiples veces y sacar los valores medios. En caso de querer probar una única vez, se deja en 1

currentIteration = 0 # Valor actual de la iteración

#Variables auxiliares
infoFitnesses = []
infoTimes = []

#Función Rastrigin para evaluación de fitness
def rastrigin(x):
    fitness=0
    a=10
    for i in range(len(x)):
        fitness+=((x[i]**2)-(a*(math.cos(2*x[i]*math.pi))))
    fitness+=(a*len(x))
    return fitness

#Función Ackley para evaluación de fitness
def ackley(solution):
    x1 = solution[0]
    x2 = solution[1]

    fitness = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x1**2 + x2**2))) - math.exp(0.5 * (math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2))) + 20 + math.e
    return fitness

#Función Esfera para evaluación de fitness
def sphere(solution):
    x1 = solution[0]
    x2 = solution[1]
    fitness = (x1**2) + (x2**2)
    return fitness

#Función que ejecuta el algoritmo CEM Original
def CEM(cost_func, bounds, popsize, elitesize, generations):
    global start
    start = time.time()
    boundsArr = np.array([bounds])
    sigma = np.std(boundsArr) / 3
    mu = np.mean(boundsArr)

    bestFitness = 100000
    generationBestFitness = 0
    countFromBestFitness = 0

    for z in range(generations):
        print("GENERATION: ", z)
        population = []
        s1 = np.random.normal(mu, sigma, popsize)
        s2 = np.random.normal(mu, sigma, popsize)
        for i in range(popsize):
            point = [s1[i], s2[i]]
            population.append(point)

        # print(population)
        results = []
        for j in range(len(population)):
            fitness = cost_func(population[j])
            objResult = {
                "point": population[j],
                "fitness": fitness
            }
            results.append(objResult)
        
        # results.sort(reverse=True)
        results.sort(key = lambda p: p["fitness"])
        eliteResults = results[:elitesize]

        elitePoints = []
        for j in range(len(eliteResults)):
            elitePoints.append(eliteResults[j]["point"])

        print(elitePoints)

        newBoundsArr = np.array(elitePoints)

        mu = np.mean(newBoundsArr)
        sigma = np.std(newBoundsArr) / 3

        if abs(mu) < abs(bestFitness):
            bestFitness = mu 
            generationBestFitness = z+1
            countFromBestFitness = 0
        else:
            countFromBestFitness += 1
        
        timeNow = time.time()
        currentTime = timeNow-start
        
        infoFitnesses[z]["iteration" + str(currentIteration+1)] = bestFitness
        infoTimes[z]["iteration" + str(currentIteration+1)] = currentTime

for i in range(maxGen):
    infoFitnesses.append({
        "currentGen": i+1
    })
    infoTimes.append({
        "currentGen": i+1
    })

#Ciclo de ejecución de todas las iteraciones configuradas
while (currentIteration < maxIter):
    start = 0
    end = 0

    # Quitar el comentario de la función que se desea ejecutar

    CEM(sphere, [(-5, 5), (5, 5), (5, -5), (-5, -5)], 100, 5, 30)
    # CEM(ackley, [(-5, 5), (5, 5), (5, -5), (-5, -5)], 100, 5, 30)
    # CEM(rastrigin, [(-5.12, 5.12), (5.12, 5.12), (5.12, -5.12), (-5.12, -5.12)], 100, 5, 30)
    currentIteration += 1


#Configuración para la generación de informes
fieldnames = ['currentGen']

for i in range(maxIter):
    fieldnames.append('iteration'+str(i+1))

fieldnames.append('mean')

infoResumen = []

arrayGen = []
arrayFitness = []
arrayTime = []

for i in range(maxGen):
    sumGenFit = 0
    totalGenFit = 0
    sumGenTime = 0
    totalGenTime = 0
    for x in range(maxIter):
        sumGenFit += infoFitnesses[i]["iteration"+str(x+1)]
        sumGenTime += infoTimes[i]["iteration"+str(x+1)]
        totalGenFit += 1
        totalGenTime += 1
        
    meanFitness = sumGenFit / totalGenFit
    meanTime = sumGenTime / totalGenTime
    infoFitnesses[i]["mean"] = meanFitness
    infoTimes[i]["mean"] = meanTime

    infoResumen.append({
        "currentGen": i,
        "meanFitness": meanFitness,
        "meanTime": meanTime
    })

    if i > 0:
        arrayGen.append(i)
        arrayFitness.append(meanFitness)
        arrayTime.append(meanTime)

fieldnamesResume = ['currentGen', 'meanFitness', 'meanTime']

with open('infoCEMRastriginFitnesses.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(infoFitnesses)

with open('infoCEMRastriginTimes.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(infoTimes)

with open('ResumenCEMRastrigin.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnamesResume)
    writer.writeheader()
    writer.writerows(infoResumen)

plt.rc('xtick',labelsize=4)
plt.rc('ytick',labelsize=4)

ax = plt.axes(projection='3d')

x = np.array(arrayTime)
y = np.array(arrayFitness)
z = np.array(arrayGen)

ax.plot_trisurf(x, y, z, cmap=plt.cm.winter, linewidth=0, antialiased=True, label="Normal")

ax.set_zticks(np.arange(0, maxGen+1))

ax.set_title("CEM - Rastrigin")
ax.ticklabel_format(axis='y', style='sci', useOffset=False)
ax.set_xlabel("Tiempo")
ax.set_ylabel("Fitness")
ax.set_zlabel("Generación", rotation=90)
ax.set_autoscalex_on(True)
ax.set_autoscaley_on(True)
ax.set_autoscalez_on(True)
ax.zaxis.set_rotate_label(False)

ax.view_init(elev=10, azim=-175)

# plt.show()
plt.savefig("GraficoCEMRastrigin.png", dpi=1600)



        