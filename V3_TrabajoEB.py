# Mejora implementada, en vez de cargar una y otra vez en memoria el archivo, los datos se leen una sola vez

from numba import prange
from mpi4py import MPI
from os import system
import numpy as np
import numba
import math
import time
import sys


numba.config.THREADING_LAYER = 'omp'

@numba.njit(parallel=True)
def calculaVecinosCercanos(matriz, vector, k, candidatos, actual):
    N = len(vector)
    M = actual
    # Calcular las distancias euclideas
    for i in prange(M):
        d_euclid = 0
        for j in prange(N):
            d_euclid += math.pow((matriz[i][j] - vector[j]), 2)
        d_euclid = math.sqrt(d_euclid)

        # Guardar la distancia euclidea y el indice del dia posterior
        for q in range(k):
            if candidatos[q][0] == -1 or candidatos[q][0] > d_euclid:
                candidatos[q][0] = d_euclid
                candidatos[q][1] = i + 1
                break

@numba.njit(parallel = False)
def media(lista_de_vectores, prediccion):
    M = len(lista_de_vectores)
    N = len(prediccion)
    for i in prange(M):
        for j in prange(N):
            prediccion[j] += float(lista_de_vectores[i][j])

    for j in range(N):
        prediccion[j] = float(prediccion[j]) / float(M)


@numba.njit()
def MAPE(Pn, Rn):
    h = len(Pn)
    mape = 0
    for i in prange(h):
        mape = abs(Rn[i] - Pn[i]) / abs(Rn[i])
    return mape * (100 / h)


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Variables globales -----------------------------------------------------------------------------------------------
    if len(sys.argv) == 3:
        numhilos = int(sys.argv[1])
        nomarchivo = sys.argv[2]
    else:
        numhilos = 1
        nomarchivo = "datos_1X.txt"
    k_vecinos = 2
    numba.set_num_threads(numhilos)

    with open(nomarchivo, 'r') as fp:
        dimensiones = list(map(int, fp.readline().split()))
    fils = dimensiones[0]
    cols = dimensiones[1]

    # Carga de todos los datos en la memoria

    globalData = []
    print("[" + str(rank) + "] copiando a memoria el archivo...")
    with open(nomarchivo, 'r') as fp:
        fp.readline()
        for line in fp:
            globalData.append(list(map(float, line.strip().split(','))))
    globalData = np.array(globalData)
    print("[" + str(rank) + "] copiado")
    comm.barrier()
    # ------------------------------------------------------------------------------------------------------------------

    if rank == 0:
        # Root lee los datos que se usan para predecir y los distribuye al resto de procesos ---------------------------
        system("rm -r Resultados")

        init = time.perf_counter()
        print("start program")
        start_index_predicciones = fils - 1 - 1000
        indices = [x for x in range(fils)]
        indices = indices[start_index_predicciones:]

        datosReales = globalData[len(globalData) - 1000:]
        datosParaPredecir = globalData[len(globalData) - 1001: len(globalData) - 1]

        if size > 1:
            datosParaPredecir = np.array_split(datosParaPredecir, size)
            indices = np.array_split(indices, size)
        # --------------------------------------------------------------------------------------------------------------
    else:
        datosParaPredecir = None
        indices = None

    # En la region distribuida reparto los datos a predecir ------------------------------------------------------------
    if size > 1:
        datosParaPredecirLocal = comm.scatter(datosParaPredecir, root=0)
        indicesLocal = comm.scatter(indices, root=0)
    else:
        datosParaPredecirLocal = datosParaPredecir
        indicesLocal = indices

    # Inicio: Calculo de las predicciones ------------------------------------------------------------------------------
    prediccionesLocales = []
    for i in range(len(datosParaPredecirLocal)):
        vectorDiaActual = np.array(datosParaPredecirLocal[i])
        indiceDiaActual = indicesLocal[i]
        # print(rank, "- prediciendo para el dia", indiceDiaActual, "progreso:", i, "/", len(datosParaPredecirLocal))

        vecinos = np.array([[-1, 0] for _ in range(k_vecinos)])
        prediccion = np.array([0 for _ in range(cols)])

        calculaVecinosCercanos(globalData, vectorDiaActual, k_vecinos, vecinos, indiceDiaActual)

        lista_vecinos = []
        # Cuando hemos encontrado los vecinos buenos los copiamos en un vector
        for index in vecinos:
            lista_vecinos.append(globalData[index[1]])
        lista_vecinos = np.array(lista_vecinos)
        media(lista_vecinos, prediccion)
        prediccionesLocales.append(prediccion)
    # FIN: Realizar predicciones----------------------------------------------------------------------------------------
    if rank != 0:
        # Si hubiese mas de un proceso, este envía sus resultados a root -----------------------------------------------
        comm.send(prediccionesLocales, dest=0)

    if rank == 0:
        prediccionesGlobales = [prediccionesLocales, ]
        for i in range(1, size):
            prediccionesGlobales.append(comm.recv(source=i))  # empiezo a recibir las predLocales en el orden correcto
        prediccionesGlobales = [item for sublist in prediccionesGlobales for item in sublist]

        # Hora de escribir los archivos
        system("mkdir Resultados")
        f1 = open("Resultados/predicciones.txt", 'x')
        f2 = open("Resultados/MAPE.txt", 'x')
        f3 = open("Resultados/tiempo.txt", 'x')

        # Escribir las predicciones
        for pred in prediccionesGlobales:
            f1.write(str(list(pred)).lstrip("[").rstrip(']') + "\n")

        # Escribir los MAPEs
        for pred, real in zip(prediccionesGlobales, datosReales):
            pred = np.array(pred)
            real = np.array(real)
            mape = MAPE(pred, real)
            f2.write(str(mape) + "\n")

        # Escribir el tiempo
        print("End program")
        tiempo = nomarchivo + "\nTiempo total:" + str(time.perf_counter() - init) + "segundos\n"
        f3.write(tiempo)
        f1.close()
        f2.close()
        f3.close()

        system("cat Resultados/tiempo.txt")
