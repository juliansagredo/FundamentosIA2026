#=================================
#  Laberinto
#=================================
#  Julián T. Sagredo
#  Fundamentos de IA
#  ESFM IPN Junio 2025
#=================================

#============================================
#  De pyamaze.py en el directorio laberinto
#============================================
from laberinto.pyamaze import maze,agent,COLOR

#==========================
#  Crear laberinto 
#  x,y posición de la meta
#==========================
m=maze(30,50)
m.CreateMaze(x=1,y=20)

#===============================
#  Poner agente en el laberinto 
#===============================
a=agent(m,footprints=True,filled=True)

#====================================
#  Graficar trayectoria del agente
#====================================
m.tracePath({a:m.path},delay=5)
m.run()
