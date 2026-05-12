#==============================================================================
#  Red Neuronal Feed-Forward en JAX (Sin TensorFlow)
#==============================================================================
#  Julián T. Becerra Sagredo 
#  ESFM IPN - Marzo 2026
#  Objetivo: Clasificar dígitos escritos a mano (MNIST) desde cero.
#==============================================================================

import jax
import jax.numpy as jnp
from jax import grad, jit, random
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

#==============================================================================
# 1. Configuración de Hiperparámetros (Las "perillas" de control)
#==============================================================================
# Estructura: 784 entradas (28x28 píxeles) 
#            -> dos capas ocultas 
#            -> 10 salidas (números 0-9)
#==============================================================================
layer_sizes = [784, 512, 256, 10]
step_size = 0.01    # Tasa de aprendizaje: qué tan rápido ajustamos los pesos
num_epochs = 10     # Cuántas veces la red verá todo el conjunto de datos
batch_size = 128    # Cuántas imágenes procesamos al mismo tiempo

#==============================================================================
# 2. Arquitectura de la Red (El "cerebro" matemático)
#==============================================================================
def init_network_params(sizes, key):
    """
    =============================================================
    Crea los pesos (w) y sesgos (b) iniciales de forma aleatoria.
    En JAX, la aleatoriedad es explícita mediante llaves (keys).
    =============================================================
    """
    keys = random.split(key, len(sizes))
    #==============================================================================
    # Usamos inicialización 'He' (sqrt(2/m)) para que las señales no mueran en ReLu
    #==============================================================================
    return [(random.normal(k, (m, n)) * jnp.sqrt(2/m), jnp.zeros(n))
            for k, m, n in zip(keys, sizes[:-1], sizes[1:])]


#=======================================
#  Arquitectura de la red Forward Pass
#=======================================
def predict(params, inputs):
    """
    ==================================================
    Función de 'Pase hacia adelante' (Forward pass).
    Toma una imagen y nos dice qué número cree que es.
    ==================================================
    """
    #====================================================================
    # Convertimos la imagen 2D (28, 28) en una fila larga de 784 píxeles
    #====================================================================
    activations = inputs.reshape((inputs.shape[0], -1))
    
    #========================================
    # Procesamos cada capa excepto la última
    #========================================
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        #==================================================================
        # ReLu: Deja pasar los positivos y convierte los negativos en cero
        #==================================================================
        activations = jax.nn.relu(outputs)
    
    #============================================================================
    # Capa final: Genera los 'logits' (puntuaciones antes de ser probabilidades)
    #============================================================================
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    
    #=============================================================================
    # Log-Softmax: Normaliza los resultados para que el entrenamiento sea estable
    #              Convierte resultados en probabilidades para toma de decisiones
    #=============================================================================
    return logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)

#========================================
# 3. Uso de la Red y Función de Pérdida
#========================================
def loss(params, batch):
    """
    ================================================================
    Calcula el error. Comparamos la predicción con la etiqueta real.
    ================================================================
    """
    inputs, targets = batch
    preds = predict(params, inputs)
    #=======================================================
    # Cross-entropy: Error en el espacio de probabilidades  
    #=======================================================
    return -jnp.mean(preds * targets)

#=================================
#  Nuevos valores para los pesos 
#================================
@jit
def update(params, batch):
    """
    ====================================================================
    Ajusta los parámetros usando el Gradiente
    @jit: Compila esta función para que corra a velocidad máxima (XLA).
    ====================================================================
    """
    #====================================================================
    # grad() obtiene la derivada de la pérdida respecto a los parámetros
    #====================================================================
    grads = grad(loss)(params, batch)
    
    #===========================================================
    # Nuevos pesos en dirección contraria al gradiente de error
    #===========================================================
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

#================================================================
# 4. Manejo de Datos (Usando Scikit-Learn como 'mesero' de datos)
#================================================================
def get_datasets():
    """ 
    ===================================================================
    Carga MNIST usando OpenML, normaliza píxeles a [0, 1],
    prepara etiquetas One-Hot y divide en Train / Test.
    ===================================================================
    """
    print("Descargando MNIST vía OpenML (esto puede tardar unos segundos la primera vez)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist["data"], mnist["target"].astype(np.int64)
    
    #=======================================================
    # Transformamos: 255 (blanco) -> 1.0, 0 (negro) -> 0.0
    #=======================================================
    X = X / 255.0
    
    #=============================================
    # Convertimos etiquetas a formato One-Hot
    #=============================================
    y_one_hot = np.eye(10)[y]
    
    #========================================================================
    # Dividimos en Train (60,000) y Test (10,000) estándar de MNIST
    #========================================================================
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y_one_hot[:60000], y_one_hot[60000:]
    
    #====================================================================
    # Empaquetamos en lotes (batches) para JAX
    #====================================================================
    train_batches = []
    for i in range(0, len(X_train), batch_size):
        batch_X = jnp.array(X_train[i:i+batch_size])
        batch_y = jnp.array(y_train[i:i+batch_size])
        train_batches.append((batch_X, batch_y))
        
    return train_batches, X_test, y_test

#==============================================================================
# 5. Visualización y Entrenamiento
#==============================================================================
def visualizar_predicciones(params, inputs, targets, num_imagenes=10):
    #=====================================================================
    """ Muestra imágenes reales con la etiqueta que la red les asignó """
    #=====================================================================
    logits = predict(params, inputs)
    
    #================================================================
    # argmax nos da el índice del valor más alto (el número predicho)
    #================================================================
    predicciones = jnp.argmax(logits, axis=1)
    etiquetas_reales = jnp.argmax(targets, axis=1)

    plt.figure(figsize=(15, 3))
    for i in range(num_imagenes):
        plt.subplot(1, num_imagenes, i + 1)
        plt.imshow(inputs[i].reshape(28, 28), cmap='gray')
        
        #================================
        # Verde si acertó, rojo si falló
        #================================
        color = 'green' if predicciones[i] == etiquetas_reales[i] else 'red'
        plt.title(f"IA dice: {predicciones[i]}\nReal: {etiquetas_reales[i]}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#============================
# --- Inicio del proceso ---
#============================
key = random.PRNGKey(42) # Semilla de aleatoriedad para reproducir resultados
params = init_network_params(layer_sizes, key)
train_batches, X_test_full, y_test_full = get_datasets()

#=================
#  ENTRENAMIENTO
#=================
print("Iniciando entrenamiento en GPU...")
for epoch in range(num_epochs):
    for batch in train_batches:
        params = update(params, batch)
    print(f"Época {epoch + 1}/{num_epochs} finalizada")

#===================
# Evaluación final
#===================
print("\nVisualizando resultados finales...")
# Tomamos los primeros 10 elementos del conjunto de pruebas
test_inputs = jnp.array(X_test_full[:10])
test_targets = jnp.array(y_test_full[:10])

visualizar_predicciones(params, test_inputs, test_targets)
