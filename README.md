# Sistemas Distribuidos: Trabajo de EB
## Enunciado
Se plantea implementar una solución simple basada en encontrar los k vecinos más cercanos. Para hacer más sencilla la comprensión, se va a detallar a partir de un ejemplo. Suponga que se dispone de una serie temporal que tiene mediciones horarias, esto es, para cada día se tienen 24 mediciones. El objetivo del algoritmo será proporcionar los 24 valores (h=24, siendo h el horizonte de predicción) correspondientes al día siguiente. Para realizar dicha predicción, se va a buscar en el histórico de datos aquellos días que más se parezcan al día actual y, una vez localizados, mirar qué pasó el día posterior (recuperar los h valores siguientes de los días más parecidos). Esto es, se van a buscar los k vecinos más cercanos al último día conocido, siendo k un valor que deberá proporcionar el usuario. Posteriormente, se realizará una media aritmética de los h valores posteriores de cada uno de esos k días y esa será, justamente,
la predicción realizada.

Existen múltiples métricas para evaluar la calidad de las predicciones, siendo cada una de ellas específica para
según qué tarea. En esta actividad evaluable, se solicita utilizar el Error Porcentual Absoluto Medio,

## Dependencias del proyecto
Para poder ejecutar el proyecto asegurese de que el/los equipos donde vaya a ejecutar el código tienen instalados los siguientes requisitos:

- Versión del interprete Python: 3.6 o superior
- Comando pip para instalar paquetes
- Tener habilitado la omp como capa de paralelismo

Dependencias de del proyecto (paquetes de python utilizados):

- #### mpi4py
  Si ya tiene instalado en el/los equipos el compilador de MPI (mpicc) entonces puede ejecutar el siguiente:
  
  ```python -m pip install mpi4py```
  
- #### numba
  `pip install numba`

## Ejecución
El siguiente comando sirve para ejecutar el proyecto

`mpiexec -np <numero de procesos> --machinefile <machinefile> python V3_TrabajoEB.py <numhilos> <archivo>`

