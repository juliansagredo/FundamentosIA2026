#==============================================================================
#  Red Neuronal Feed-Forward en JAX (Representación de Embeddings)
#==============================================================================
#  Julián T. Becerra Sagredo 
#  ESFM IPN - Marzo 2026
#  Objetivo: Visualizar cómo la IA "ve" los números internamente.
#==============================================================================

import jax
import jax.numpy as jnp
from jax import grad, jit, random
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

#==============================================================================
# 1. Configuración de Hiperparámetros
#==============================================================================
layer_sizes = [784, 512, 256, 10]
step_size = 0.01
num_epochs = 10
batch_size = 128

#==============================================================================
# 2. Arquitectura de la Red
#==============================================================================
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [(random.normal(k, (m, n)) * jnp.sqrt(2/m), jnp.zeros(n))
            for k, m, n in zip(keys, sizes[:-1], sizes[1:])]

def predict(params, inputs):
    activations = inputs.reshape((inputs.shape[0], -1))
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jax.nn.relu(outputs)
    
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)

#========================================
# 3. Función de Pérdida y Actualización
#========================================
def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(preds * targets)

@jit
def update(params, batch):
    grads = grad(loss)(params, batch)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

#================================================================
# 4. Manejo de Datos (Carga limpia sin TensorFlow)
#================================================================
def get_datasets():
    print("Cargando MNIST vía OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist["data"], mnist["target"].astype(np.int64)
    
    X = X / 255.0
    y_one_hot = np.eye(10)[y]
    
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y_one_hot[:60000], y_one_hot[60000:]
    
    train_batches = []
    for i in range(0, len(X_train), batch_size):
        batch_X = jnp.array(X_train[i:i+batch_size])
        batch_y = jnp.array(y_train[i:i+batch_size])
        train_batches.append((batch_X, batch_y))
        
    return train_batches, X_test, y_test

#==============================================================================
# 5. Funciones de Extracción de Embeddings y Visualización
#==============================================================================
def extraer_embeddings(params, inputs):
    """ Devuelve el vector latente de la capa de 256 dimensiones """
    activations = inputs.reshape((inputs.shape[0], -1))
    # Procesamos solo las capas ocultas (todas menos la última)
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jax.nn.relu(outputs)
    return activations

def visualizar_embeddings(params, X_test, y_test, index=0):
    """ Muestra la entrada original y su embedding como matriz 16x16 """
    # Preparamos una sola imagen para la red
    input_img = jnp.array([X_test[index]])
    target_val = y_test[index]

    # Extraemos embedding y predicción
    embedding = extraer_embeddings(params, input_img)[0]
    logits = predict(params, input_img)
    prediccion = jnp.argmax(logits[0])
    etiqueta_real = jnp.argmax(target_val)

    plt.figure(figsize=(10, 4))

    # Subplot 1: Imagen Original
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"Entrada (Real: {etiqueta_real} | IA: {prediccion})")
    plt.axis('off')

    # Subplot 2: El Embedding (16x16 = 256 dimensiones)
    plt.subplot(1, 2, 2)
    plt.imshow(embedding.reshape(16, 16), cmap='magma')
    plt.colorbar(label="Activación ReLU")
    plt.title("Embedding (Capa 256 dim)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

#============================
# --- Inicio del proceso ---
#============================
key = random.PRNGKey(42)
params = init_network_params(layer_sizes, key)
train_batches, X_test, y_test = get_datasets()

print("Entrenando en GPU...")
for epoch in range(num_epochs):
    for batch in train_batches:
        params = update(params, batch)
    print(f"Época {epoch + 1}/{num_epochs} finalizada")

print("\nMostrando la representación interna (Embeddings)...")
# Visualizamos los primeros 5 ejemplos del set de prueba
for i in range(5):
    visualizar_embeddings(params, X_test, y_test, index=i)
