import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [10, 5]

class timePlotting():	
	tiempo_1X_hilos = [
	[1,2,3,4],
	[0.9419068719998904,
	0.8851777159998164,
	0.8655707070001881,
	0.8636832570000479]
	]
	
	tiempo_1X_procesos = [
	[1,2,3,4],
	[0.9300465220003389,
	0.9383696139998392,
	1.0009819269998843,
	1.057404205000239]
	]
	
	tiempo_10X_hilos = [
	[1,2,3,4],
	[1.9400195670000357,
	1.3976429630001803,
	1.2213310429997364,
	1.1271094869998706]
	]
	
	tiempo_10X_procesos = [
	[1,2,3,4],
	[1.9400195670000357,
	1.4557165020000866,
	1.3107672270002695,
	1.2960262329997931]
	]
	
	tiempo_1000X_hilos = [
	[1,2,3,4],
	[123.78594803099986,
	63.35208489200022,
	45.57019119100005,
	35.55413338700009]
	]
	
	tiempo_1000X_procesos = [
	[1,2,3,4],
	[123.78594803099986,
	65.05298386899995,
	46.78402788400035,
	39.17807833300003]
	]
	
	def plotData(self, data, string):
		x = np.array(data[0], dtype=np.int)
		y = np.array(data[1])
		plt.xlabel("Numero de " + string + " utilizados")
		plt.ylabel("Tiempo en segundos")
		plt.plot(x,y,color='red', marker="o")
		plt.bar(x,y, width=0.5)
		plt.show()
	
