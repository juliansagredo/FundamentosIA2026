#==============================================================================
#  Red Neuronal Feed-Forward en JAX (Mapa topológico con t-SNE)
#==============================================================================
#  Julián T. Becerra Sagredo 
#  ESFM IPN - Marzo 2026
#  Objetivo: Proyectar el espacio latente de 256D a 2D para ver grupos.
#==============================================================================

import jax
import jax.numpy as jnp
from jax import grad, jit, random
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
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
# 5. Funciones de Embeddings y t-SNE
#==============================================================================
def extraer_embeddings(params, inputs):
    activations = inputs.reshape((inputs.shape[0], -1))
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jax.nn.relu(outputs)
    return activations

def visualizar_tsne(params, X_test, y_test, num_muestras=1000):
    """ Crea el mapa 2D de los embeddings """
    print(f"Calculando t-SNE para {num_muestras} muestras...")
    
    # Tomamos una muestra del conjunto de prueba
    x_sample = jnp.array(X_test[:num_muestras])
    y_sample = np.argmax(y_test[:num_muestras], axis=1)
    
    # Extraemos los vectores latentes (256D)
    embeddings = extraer_embeddings(params, x_sample)
    
    # Proyectamos a 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Graficamos
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=y_sample, cmap='tab10', alpha=0.7, s=15)
    plt.colorbar(scatter, ticks=range(10), label="Dígito")
    plt.title("Mapa Topológico de Embeddings (t-SNE)\nProyección de 256D a 2D")
    plt.grid(True, linestyle='--', alpha=0.5)
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

print("\nGenerando mapa topológico...")
visualizar_tsne(params, X_test, y_test, num_muestras=2000)
