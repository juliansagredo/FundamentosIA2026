#===========================
#  Gráficas de aprendizaje
#===========================
import matplotlib.pyplot as plt
import warnings

# Detectar si estamos en un entorno interactivo (Jupyter/IPython)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from IPython import display
        is_ipython = True
    except ImportError:
        is_ipython = False

#==================================
#  Gráfica dinámica en el tiempo
#==================================
def graficar(scores, mean_scores):
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    plt.ion()  # Modo interactivo
    plt.clf()
    plt.title('Entrenando...')
    plt.xlabel('Número de Juegos')
    plt.ylabel('Puntaje')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
