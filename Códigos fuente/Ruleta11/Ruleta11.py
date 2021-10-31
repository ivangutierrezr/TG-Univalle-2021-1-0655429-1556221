import numpy as np
import networkx as nx
from math import log, sqrt
import time
import csv

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

#Variables iniciales
start = 0 # Variables para medición de tiempo
end = 0 # Variables para medición de tiempo
current_gen = 0 # Generación actual
pop = 100 # Tamaño de población inicial
mu, sigma = 0, 0.4 # Media y desviación estándar
max_gen = 30 # Máximo de generaciones
chosen_size = 5 # Tamaño de la población a escoger para reproducción
key_count = 100 # Variable para controlar el último valor de llave creado. Esta variable es para generar la red en NetworkX y el valor de entrada es igual al de la variable pop
max_iter = 100 # Número máximo de iteraciones para cada ciclo generacional. Se usa para correr el algorítmo múltiples veces y sacar los valores medios. En caso de querer probar una única vez, se deja en 1
current_iteration = 0 # Valor actual de la iteración

#Variables auxiliares
universe1 = []
universe2 = []
universe3 = []
universe4 = []
chosenUniverse1 = []
chosenUniverse2 = []
chosenUniverse3 = []
chosenUniverse4 = []
notChosenUniverse1 = []
notChosenUniverse2 = []
notChosenUniverse3 = []
notChosenUniverse4 = []
countReplacedU1 = 0
countReplacedU2 = 0
countReplacedU3 = 0
countReplacedU4 = 0

infoFitnesses = [{
    "currentGen": 0
}]
infoTimes = [{
    "currentGen": 0
}]

#función auxiliar para volver las variables auxiliares a su estado original
def resetStates():
    global chosenUniverse1
    global chosenUniverse2
    global chosenUniverse3
    global chosenUniverse4
    global notChosenUniverse1
    global notChosenUniverse2
    global notChosenUniverse3
    global notChosenUniverse4
    global countReplacedU1
    global countReplacedU2
    global countReplacedU3
    global countReplacedU4
    chosenUniverse1 = []
    chosenUniverse2 = []
    chosenUniverse3 = []
    chosenUniverse4 = []
    notChosenUniverse1 = []
    notChosenUniverse2 = []
    notChosenUniverse3 = []
    notChosenUniverse4 = []
    countReplacedU1 = 0
    countReplacedU2 = 0
    countReplacedU3 = 0
    countReplacedU4 = 0
    return True

#Función esfera para evaluación de fitness
def getFitnessBySphere(x, y):
    fitness = (x**2) + (y**2)
    return fitness

#función para retornar el individuo con mejor fitness
def getBestFitness():
    fullUniversesData = universe1 + universe2 + universe3 + universe4
    dataUniverses = []
    for i in range(len(fullUniversesData)):
        dataUniverses.append(fullUniversesData[i]["fitness"])
    
    dataUniverses.sort()
    return dataUniverses[0]

#Función de entropía cruzada logarítmica entre cada individuo de los universos. Esta función sirve para medir la entropía entre dos univesos previo a su cruce
def cross_entropy(p, q):
	return -sum([abs(p[i])*log(abs(q[i])) for i in range(len(p))])

#Función para evaluar la entropía cruzada logarítmica entre dos universos
def evaluateCEL(u1, u2):
    results = []
    for i in range(chosen_size):
        # Sen calcula la entropía cruzada entre dos universos para cada individuo
        try:
            ce = cross_entropy(u1[i]["coords"], u2[i]["coords"])
            results.append(ce)
        except:
            print("")

    # Se calcula la media de entropía cruzada entre dos universos
    mean_ce = np.mean(results)
    return mean_ce

#Función que se encarga de reproducir el universo. En este caso reproduce los individuos de mejor fitness organizados en orden ascendente de un universo con los de mejor fitness de otro universo organizados aleatoriamente
def reproduceUniverses(u1, u2):
    childs = []
    listOFNumbers = np.arange(0, chosen_size, 1)
    np.random.shuffle(listOFNumbers)
    newListOFNumbers = listOFNumbers.tolist()
    for i in range(chosen_size):
        # Se cruzan los individuos de los dos universos de manera aleatoria. Se genera una lista de números aleatorios del 0 al valor de la variable chosen_size pero organizados de manera aleatoria, de modo que el primer elemento de el universo u1 se mezclará con el elemento del universo u2 correspondiente a la primera posicion de la lista aleatoria
        randomPos = newListOFNumbers[i]
        # Cada nuevo hijo se generará a partir de la media entre los padres
        newX = (u1[i]["coords"][0] + u2[randomPos]["coords"][0]) / 2
        newY = (u1[i]["coords"][1] + u2[randomPos]["coords"][1]) / 2
        fitness = getFitnessBySphere(newX, newY)
        newChild = {
            "fitness": fitness,
            "coords": [newX, newY],
            "parents": [u1[i]["keyP"], u2[randomPos]["keyP"]]
        }
        childs.append(newChild)
    return childs

# Función encargada de añadir los nuevos padres al nodo durante la función de reemplazo
def replaceParentsOnNode(currentParents, newParents):
    foundParents = currentParents
    for i in range(len(newParents)):
        foundParent = False
        try:
            pos = currentParents.index(newParents[i])
            foundParent = True
        except:
            foundParents.append(newParents[i])

    return foundParents

#Función de reemplazo: Recibe todos los nodos nuevos creados luego de la función de cruce y reemplaza los de peor fitness de cada universo por cada uno de ellos. Se evalúa según su posición frente al punto medio para saber a que universo corresponde el nuevo nodo.
def replaceNodesOnNoChosen(childs):
    global universe1
    global universe2
    global universe3
    global universe4
    global chosenUniverse1
    global chosenUniverse2
    global chosenUniverse3
    global chosenUniverse4
    global notChosenUniverse1
    global notChosenUniverse2
    global notChosenUniverse3
    global notChosenUniverse4
    global countReplacedU1
    global countReplacedU2
    global countReplacedU3
    global countReplacedU4
    global end
    global infoFitnesses
    global infoFitnesses
    global infoTimes
    global key_count
    # Se utiliza técnica de reemplazo directo, cada nuevo nodo reemplazará a un nodo no escogido en la técnica elitista
    for i in range(len(childs)):
        x = childs[i]["coords"][0]
        y = childs[i]["coords"][1]
        fitness = childs[i]["fitness"]
        parents = childs[i]["parents"]
        point = [x, y]
        objChild = {
            "keyP": 0,
            "fitness": fitness,
            "coords": point,
            "parents": parents
        }
        if x >= mu and y >= mu:
            posToReplace = len(notChosenUniverse1)-1-countReplacedU1
            if posToReplace >= 0:
                notChosenUniverse1[posToReplace]["fitness"] = fitness
                notChosenUniverse1[posToReplace]["coords"] = point
                notChosenUniverse1[posToReplace]["parents"] = replaceParentsOnNode(notChosenUniverse1[posToReplace]["parents"], parents)
            else:
                key_count = key_count+1
                objChild["keyP"] = key_count
                notChosenUniverse1.append(objChild)
            countReplacedU1 = countReplacedU1+1

        elif x >= mu and y <= mu:
            posToReplace = len(notChosenUniverse2)-1-countReplacedU2
            if posToReplace >= 0:
                notChosenUniverse2[posToReplace]["fitness"] = fitness
                notChosenUniverse2[posToReplace]["coords"] = point
                notChosenUniverse2[posToReplace]["parents"] = replaceParentsOnNode(notChosenUniverse2[posToReplace]["parents"], parents)
            else:
                key_count = key_count+1
                objChild["keyP"] = key_count
                notChosenUniverse2.append(objChild)
            countReplacedU2 = countReplacedU2+1

        elif x <= mu and y <= mu:
            posToReplace = len(notChosenUniverse3)-1-countReplacedU3
            if posToReplace >= 0:
                notChosenUniverse3[posToReplace]["fitness"] = fitness
                notChosenUniverse3[posToReplace]["coords"] = point
                notChosenUniverse3[posToReplace]["parents"] = replaceParentsOnNode(notChosenUniverse3[posToReplace]["parents"], parents)
            else:
                key_count = key_count+1
                objChild["keyP"] = key_count
                notChosenUniverse3.append(objChild)
            countReplacedU3 = countReplacedU3+1

        else:
            posToReplace = len(notChosenUniverse4)-1-countReplacedU4
            if posToReplace >= 0:
                notChosenUniverse4[posToReplace]["fitness"] = fitness
                notChosenUniverse4[posToReplace]["coords"] = point
                notChosenUniverse4[posToReplace]["parents"] = replaceParentsOnNode(notChosenUniverse4[posToReplace]["parents"], parents)
            else:
                key_count = key_count+1
                objChild["keyP"] = key_count
                notChosenUniverse4.append(objChild)
            countReplacedU4 = countReplacedU4+1

    universe1 = chosenUniverse1 + notChosenUniverse1
    universe2 = chosenUniverse2 + notChosenUniverse2
    universe3 = chosenUniverse3 + notChosenUniverse3
    universe4 = chosenUniverse4 + notChosenUniverse4

    if resetStates():
        timeNow = time.time()
        currentTime = timeNow-start
        bestFitness = getBestFitness()
        
        infoFitnesses[current_gen]["iteration" + str(current_iteration+1)] = bestFitness
        infoTimes[current_gen]["iteration" + str(current_iteration+1)] = currentTime
        if current_gen < max_gen:
            unZipElites()
        else:
            end = time.time()
            print(end - start)

# Función que se encarga de organizar la jerarquía para la reproducción de universos
def evaluateUniverses():    
    # Hay 4 universos, para el universo 1 se ha asignado un rango de 0 a 0.25, universo 2 > 0.25 y <= 0.5, universo 3 >0.5 y <= 0.75 y universo 4 > 0.75 y <= 1.
    # Se escogerá un número aleatorio entre 0 y 1, y dependiendo del resultado, se escogerá el universo correspondiente como primero en reproducirse con su mejor par, los otros dos universos se reproducirán independientemente de su entropía.
    probU = np.random.rand()
    childs1 = []
    childs2 = []
    if probU <= 0.25:
        eval12 = evaluateCEL(chosenUniverse1, chosenUniverse2)
        eval13 = evaluateCEL(chosenUniverse1, chosenUniverse3)
        eval14 = evaluateCEL(chosenUniverse1, chosenUniverse4)
        minVal = min([eval12, eval13, eval14])
        if minVal == eval12:
            childs1 = reproduceUniverses(chosenUniverse1, chosenUniverse2)
            childs2 = reproduceUniverses(chosenUniverse3, chosenUniverse4)
        elif minVal == eval13:
            childs1 = reproduceUniverses(chosenUniverse1, chosenUniverse3)
            childs2 = reproduceUniverses(chosenUniverse2, chosenUniverse4)
        else:
            childs1 = reproduceUniverses(chosenUniverse1, chosenUniverse4)
            childs2 = reproduceUniverses(chosenUniverse2, chosenUniverse3)
    elif probU > 0.25 and probU <= 0.5:
        eval21 = evaluateCEL(chosenUniverse2, chosenUniverse1)
        eval23 = evaluateCEL(chosenUniverse2, chosenUniverse3)
        eval24 = evaluateCEL(chosenUniverse2, chosenUniverse4)
        minVal = min([eval21, eval23, eval24])
        if minVal == eval21:
            childs1 = reproduceUniverses(chosenUniverse2, chosenUniverse1)
            childs2 = reproduceUniverses(chosenUniverse3, chosenUniverse4)
        elif minVal == eval23:
            childs1 = reproduceUniverses(chosenUniverse2, chosenUniverse3)
            childs2 = reproduceUniverses(chosenUniverse1, chosenUniverse4)
        else:
            childs1 = reproduceUniverses(chosenUniverse2, chosenUniverse4)
            childs2 = reproduceUniverses(chosenUniverse1, chosenUniverse3)
    elif probU > 0.5 and probU <= 0.75:
        eval31 = evaluateCEL(chosenUniverse3, chosenUniverse1)
        eval32 = evaluateCEL(chosenUniverse3, chosenUniverse2)
        eval34 = evaluateCEL(chosenUniverse3, chosenUniverse4)
        minVal = min([eval31, eval32, eval34])
        if minVal == eval31:
            childs1 = reproduceUniverses(chosenUniverse3, chosenUniverse1)
            childs2 = reproduceUniverses(chosenUniverse2, chosenUniverse4)
        elif minVal == eval32:
            childs1 = reproduceUniverses(chosenUniverse3, chosenUniverse2)
            childs2 = reproduceUniverses(chosenUniverse1, chosenUniverse4)
        else:
            childs1 = reproduceUniverses(chosenUniverse3, chosenUniverse4)
            childs2 = reproduceUniverses(chosenUniverse1, chosenUniverse2)
    else:
        eval41 = evaluateCEL(chosenUniverse4, chosenUniverse1)
        eval42 = evaluateCEL(chosenUniverse4, chosenUniverse2)
        eval43 = evaluateCEL(chosenUniverse4, chosenUniverse3)
        minVal = min([eval41, eval42, eval43])
        if minVal == eval41:
            childs1 = reproduceUniverses(chosenUniverse4, chosenUniverse1)
            childs2 = reproduceUniverses(chosenUniverse2, chosenUniverse3)
        elif minVal == eval42:
            childs1 = reproduceUniverses(chosenUniverse4, chosenUniverse2)
            childs2 = reproduceUniverses(chosenUniverse1, chosenUniverse3)
        else:
            childs1 = reproduceUniverses(chosenUniverse4, chosenUniverse3)
            childs2 = reproduceUniverses(chosenUniverse1, chosenUniverse2)
    
    childs = childs1 + childs2
    replaceNodesOnNoChosen(childs)

#Función para seleccionar la población por método ruleta
def selectByRoulette(u):
    # La forma de elegir la población para este reproducción para este universo serán aquellos individuos que se encuentren más cercanos al centro
    sumFitness = 0
    fitnesses = []
    probabilities = []
    for i in range(len(u)):
        fitness = u[i]["fitness"]
        sumFitness = sumFitness + fitness
        fitnesses.append(fitness)
    
    for i in range(len(u)):
        sumProbs = 0
        if i > 0:
            sumProbs = probabilities[i-1]
        prob = sumProbs + (fitnesses[i] / sumFitness)
        probabilities.append(prob)

    chosenUniverse = []
    notChosenUniverse = []
    
    positionChosen = []

    while len(chosenUniverse) < chosen_size:
        probR = np.random.rand()
        for i in range(len(u)):
            if i < len(u)-1:
                if probR > probabilities[i] and probR < probabilities[i+1]:
                    try:
                        pos = positionChosen.index(i)
                    except:
                        chosenUniverse.append(u[i])
                        positionChosen.append(i)
            else:
                if probR >= probabilities[i]:
                    try:
                        pos = positionChosen.index(i)
                    except:
                        chosenUniverse.append(u[i])
                        positionChosen.append(i)

    for i in range(len(u)):
        try:
            pos = positionChosen.index(i)
        except:
            notChosenUniverse.append(u[i])

    objElite = {
        "chosenUniverse": chosenUniverse,
        "notChosenUniverse": notChosenUniverse
    }
    return objElite

#Función para descomprimir los universos entre los elegidos por élite y los que no serán reproducidos
def unZipElites():
    global chosenUniverse1
    global chosenUniverse2
    global chosenUniverse3
    global chosenUniverse4
    global notChosenUniverse1
    global notChosenUniverse2
    global notChosenUniverse3
    global notChosenUniverse4
    global current_gen

    chosenUniverse1Obj = selectByRoulette(universe1)
    chosenUniverse2Obj = selectByRoulette(universe2)
    chosenUniverse3Obj = selectByRoulette(universe3)
    chosenUniverse4Obj = selectByRoulette(universe4)
    chosenUniverse1 = chosenUniverse1Obj["chosenUniverse"]
    chosenUniverse2 = chosenUniverse2Obj["chosenUniverse"]
    chosenUniverse3 = chosenUniverse3Obj["chosenUniverse"]
    chosenUniverse4 = chosenUniverse4Obj["chosenUniverse"]
    notChosenUniverse1 = chosenUniverse1Obj["notChosenUniverse"]
    notChosenUniverse2 = chosenUniverse2Obj["notChosenUniverse"]
    notChosenUniverse3 = chosenUniverse3Obj["notChosenUniverse"]
    notChosenUniverse4 = chosenUniverse4Obj["notChosenUniverse"]

    current_gen += 1
    print("GENERATION: ", current_gen)
    evaluateUniverses()

#Función que se encarga de generar una nueva población en cada iteración
def runCode():
    global universe1
    global universe2
    global universe3
    global universe4
    global mu
    global sigma
    global pop
    global start
    start = time.time()
    # Se crea la primera generación a partir de dos distribuciones aleatorias gaussianas. 
    # x corresponde a las coordenadas en X para cada punto, de igual manera la variable y.
    # Se crean parejas ordenadas a partir de aquí
    x = np.random.normal(mu, sigma, pop)
    y = np.random.normal(mu, sigma, pop)
    for i in range(pop):
        fitness = getFitnessBySphere(x[i], y[i])
        point = {
            "keyP": i+1,
            "fitness": fitness,
            "coords": [x[i], y[i]],
            "parents": []
        }
        if x[i] >= mu and y[i] >= mu:
            universe1.append(point)
        elif x[i] >= mu and y[i] <= mu:
            universe2.append(point)
        elif x[i] <= mu and y[i] <= mu:
            universe3.append(point)
        else:
            universe4.append(point)

    timeNow = time.time()
    currentTime = timeNow-start
    bestFitness = getBestFitness()

    infoFitnesses[0]["iteration"+str(current_iteration+1)] = bestFitness
    infoTimes[0]["iteration"+str(current_iteration+1)] = currentTime

    unZipElites()

# Ciclo para agregar los encabezados del informe
for i in range(max_gen):
    infoFitnesses.append({
        "currentGen": i+1
    })
    infoTimes.append({
        "currentGen": i+1
    })

# Ciclo que se encarga de correr el código hasta el número de iteraciones configurado
while (current_iteration < max_iter):
    start = 0
    end = 0
    current_gen = 0
    universe1 = []
    universe2 = []
    universe3 = []
    universe4 = []
    chosenUniverse1 = []
    chosenUniverse2 = []
    chosenUniverse3 = []
    chosenUniverse4 = []
    notChosenUniverse1 = []
    notChosenUniverse2 = []
    notChosenUniverse3 = []
    notChosenUniverse4 = []
    countReplacedU1 = 0
    countReplacedU2 = 0
    countReplacedU3 = 0
    countReplacedU4 = 0
    runCode()
    current_iteration += 1

# Generación del modelo de red para la última iteracion
fullUniverse = universe1 + universe2 + universe3 + universe4
nodes = []
edges = []
for i in range(len(fullUniverse)):
    nodes.append(fullUniverse[i]["keyP"])
    for x in range(len(fullUniverse[i]["parents"])):
        if fullUniverse[i]["keyP"] != fullUniverse[i]["parents"][x]:
            if fullUniverse[i]["keyP"] < fullUniverse[i]["parents"][x]:
                point = (fullUniverse[i]["keyP"], fullUniverse[i]["parents"][x])
            else:
                point = (fullUniverse[i]["parents"][x], fullUniverse[i]["keyP"])
            try:
                pos = edges.index(point)
            except:
                edges.append(point)

G = nx.Graph()

G.add_nodes_from(nodes)
G.add_edges_from(edges, color="red")

try:
    spl = nx.average_shortest_path_length(G)
except:
    spl = "Grafo no conectado"
try:
    clust = nx.average_clustering(G)
except:
    clust = "Grafo no conectado"

list_of_edges_count = []
for i in range(len(nodes)):
    list_of_edges_count.append(len(G.edges(nodes[i])))

print("shortest path length: ", spl)
print("clustering: ", clust)
print("degree: ", list_of_edges_count)

bins = np.arange(min(list_of_edges_count), max(list_of_edges_count) + 1, 1)

plt.hist(list_of_edges_count, bins = bins, density = True, alpha=0.5, histtype='bar', ec='black')
plt.ylabel('Density')
plt.xlabel('Edges')
plt.title('Degree distribution Ruleta 11')
figHist = plt.gcf()

figHist.savefig('DistribucionGradoRuleta11.png')

plt.clf()

template = """Información de la red estrategia Ruleta 11
Promedio camino más corto: {spl} 
Promedio coeficiente de clustering: {clust}
""" 
context = {
    "spl": spl, 
    "clust": clust,
} 

with open('networkInfo.txt','w', encoding='UTF8') as myfile:
    myfile.write(template.format(**context))

nx.write_gexf(G, "infoCEMRuleta11.gexf", version="1.2draft")

#Nombre de campos para el informe completo de Tiempo y Fitness
fieldnames = ['currentGen']

for i in range(max_iter):
    fieldnames.append('iteration'+str(i+1))

fieldnames.append('mean')

infoResumen = []

arrayGen = []
arrayFitness = []
arrayTime = []

# Organización de información para generación del informe en CSV
for i in range(max_gen+1):
    sumGenFit = 0
    totalGenFit = 0
    sumGenTime = 0
    totalGenTime = 0
    for x in range(max_iter):
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

# Creación de informe completo de Fitness
with open('infoCEMRuleta11Fitnesses.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(infoFitnesses)

# Creación de informe completo de Tiempos
with open('infoCEMRuleta11Times.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(infoTimes)

# Creación de informe completo de Resumen
with open('ResumenRuleta11.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnamesResume)
    writer.writeheader()
    writer.writerows(infoResumen)

plt.rc('xtick',labelsize=4)
plt.rc('ytick',labelsize=4)
# plt.rc('ztick',labelsize=4)

ax = plt.axes(projection='3d')

x = np.array(arrayTime)
y = np.array(arrayFitness)
z = np.array(arrayGen)

ax.plot_trisurf(x, y, z, cmap=plt.cm.winter, linewidth=0, antialiased=True, label="Normal")

ax.set_zticks(np.arange(0, max_gen+1))

ax.set_title("Estrategia Ruleta 11")
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
plt.savefig("GraficoRuleta11.png", dpi=1600)