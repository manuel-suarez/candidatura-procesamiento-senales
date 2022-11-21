# Import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from numpy.random import random, randint, randn
from zernike import RZern

# Definimos parámetros iniciales
scale_factor = 40   # Factor de escala para obtener el rango dinámico de los polinomios
order = 6           # Orden de los polinomios a generar
cart = RZern(order) # Generador de polinomios
dim = 128           # Tamaño de dimensiones de imagen
L, K = dim, dim     # Tamaño de cada imagen
num = 20000         # Tamaño conjunto de entrenamiento
num_test = 5000     # Tamaño conjunto de prueba
num_epochs = 50     # Número de épocas para el entrenamiento
latent_dim = 10      # Dimensión del espacio latente de la red VAE
# Definimos el grid para la generación de las señales
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)
# Generamos 30,000 señales variando los parámetros del polinomio a generar
Z = []
for i in range(num):
  # Variamos los parámetros para el polinomio
  c = np.random.normal(size=cart.nk)
  # Generamos polinomio
  Phi = cart.eval_grid(c, matrix=True)
  # Reemplazamos NaN con 0
  p = np.nan_to_num(Phi, False, 0)
  #p = Phi
  # Reescalamos a 0-1 (necesario para que la red calcule correctamente las métricas)
  # Verificar con el Dr. si esto sería necesario en este caso
  #p_scaled = (p - np.min(p)) / (np.max(p) - np.min(p))
  # Agregamos a conjunto de resultados
  Z.append(p)

# Desplegamos una muestra
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    # print(Z[i*3 + j])
    print(f"Señal {i*3+j}, máx: {np.max(Z[i*3+j])}, min: {np.min(Z[i*3+j])}")
    ax[i,j].imshow(Z[i*3 + j])
    ax[i,j].set_title(f"Señal {i*3+j}")

fig.tight_layout()
plt.show()

# Definimos función W para el envolvimiento de fase del polinomio
def W(p):
  return np.arctan2(np.sin(p), np.cos(p))