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

# En vez de generar las derivadas direccionales ahora aplicamos el envolvimiento de fase del polinomio
WZ = []
for img in Z:
  WZ.append(W(img))

# Desplegamos una muestra de los polinomios con envolvimiento de fase
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    print(f"Señal {i*3+j}, máx: {np.max(WZ[i*3+j])}, min: {np.min(WZ[i*3+j])}")
    ax[i,j].imshow(WZ[i*3 + j])
    ax[i,j].set_title(f"W(Señal {i*3+j})")

fig.tight_layout()
plt.show()

# Generamos las derivadas direccionales para cada imagen sobre la fase envuelta
DWx = []
DWy = []
for img in Z:
  img_dy, img_dx = np.gradient(img)
  # Aplicamos W a los gradientes para eliminar las transiciones
  DWx.append(W(img_dx))
  DWy.append(W(img_dy))

# Desplegamos una muestra de los gradientes en X
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    ax[i,j].imshow(DWx[i*3 + j])
    ax[i,j].set_title(f"DWx {i*3+j}")

fig.tight_layout()
plt.show()

# Desplegamos una muestra de los gradientes en Y
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
  for j in range(3):
    #print(Dy[i*3 + j])
    ax[i,j].imshow(DWy[i*3 + j])
    ax[i,j].set_title(f"DWy {i*3+j}")

fig.tight_layout()
plt.show()

# Arquitectura del Autoencoder variacional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Dropout, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.utils.vis_utils import plot_model

# Dimensión de la imagen de entrada (el polinomio) utilizado en el entrenamiento y pruebas
INPUT_DIM     = (128,128,1)
# Utilizamos dos canales de entrada para representar las derivadas parciales del polinomio
GRADIENT_DIM  = (128,128,2)
# Dimensión del espacio latente
LATENT_DIM    = 150
BATCH_SIZE    = 384
R_LOSS_FACTOR = 100000  # 10000
EPOCHS        = 50
INITIAL_EPOCH = 0

steps_per_epoch = num//BATCH_SIZE

# Generador de muestras del espacio latente
class Sampler(keras.Model):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def __init__(self, latent_dim, **kwargs):
    super(Sampler, self).__init__(**kwargs)
    self.latent_dim = latent_dim
    self.model = self.sampler_model()
    self.built = True

  def get_config(self):
    config = super(Sampler, self).get_config()
    config.update({"units": self.units})
    return config

  def sampler_model(self):
    '''
    input_dim is a vector in the latent (codified) space
    '''
    input_data = layers.Input(shape=self.latent_dim)
    z_mean = Dense(self.latent_dim, name="z_mean")(input_data)
    z_log_var = Dense(self.latent_dim, name="z_log_var")(input_data)

    self.batch = tf.shape(z_mean)[0]
    self.dim = tf.shape(z_mean)[1]

    epsilon = tf.keras.backend.random_normal(shape=(self.batch, self.dim))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    model = keras.Model(input_data, [z, z_mean, z_log_var])
    return model

  def call(self, inputs):
    '''
    '''
    return self.model(inputs)

  class Encoder(keras.Model):
    def __init__(self, input_dim, output_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
                 use_batch_norm=True, use_dropout=True, **kwargs):
      '''
      '''
      super(Encoder, self).__init__(**kwargs)

      self.input_dim = input_dim
      self.output_dim = output_dim
      self.encoder_conv_filters = encoder_conv_filters
      self.encoder_conv_kernel_size = encoder_conv_kernel_size
      self.encoder_conv_strides = encoder_conv_strides
      self.n_layers_encoder = len(self.encoder_conv_filters)
      self.use_batch_norm = use_batch_norm
      self.use_dropout = use_dropout

      self.model = self.encoder_model()
      self.built = True

    def get_config(self):
      config = super(Encoder, self).get_config()
      config.update({"units": self.units})
      return config

    def encoder_model(self):
      '''
      '''
      encoder_input = layers.Input(shape=self.input_dim, name='encoder')
      x = encoder_input

      for i in range(self.n_layers_encoder):
        x = Conv2D(filters=self.encoder_conv_filters[i],
                   kernel_size=self.encoder_conv_kernel_size[i],
                   strides=self.encoder_conv_strides[i],
                   padding='same',
                   name='encoder_conv_' + str(i), )(x)
        if self.use_batch_norm:
          x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
          x = Dropout(rate=0.25)(x)

      self.last_conv_size = x.shape[1:]
      x = Flatten()(x)
      encoder_output = Dense(self.output_dim)(x)
      model = keras.Model(encoder_input, encoder_output)
      return model

    def call(self, inputs):
      '''
      '''
      return self.model(inputs)

# Decodificador
  class Decoder(keras.Model):
    def __init__(self, input_dim, input_conv_dim,
                 decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
                 use_batch_norm=True, use_dropout=True, **kwargs):

      '''
      '''
      super(Decoder, self).__init__(**kwargs)

      self.input_dim = input_dim
      self.input_conv_dim = input_conv_dim

      self.decoder_conv_t_filters = decoder_conv_t_filters
      self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
      self.decoder_conv_t_strides = decoder_conv_t_strides
      self.n_layers_decoder = len(self.decoder_conv_t_filters)

      self.use_batch_norm = use_batch_norm
      self.use_dropout = use_dropout

      self.model = self.decoder_model()
      self.built = True

    def get_config(self):
      config = super(Decoder, self).get_config()
      config.update({"units": self.units})
      return config

    def decoder_model(self):
      '''
      '''
      decoder_input = layers.Input(shape=self.input_dim, name='decoder')
      x = Dense(np.prod(self.input_conv_dim))(decoder_input)
      x = Reshape(self.input_conv_dim)(x)

      for i in range(self.n_layers_decoder):
        x = Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                            kernel_size=self.decoder_conv_t_kernel_size[i],
                            strides=self.decoder_conv_t_strides[i],
                            padding='same',
                            name='decoder_conv_t_' + str(i))(x)
        if i < self.n_layers_decoder - 1:

          if self.use_batch_norm:
            x = BatchNormalization()(x)
          x = LeakyReLU()(x)
          if self.use_dropout:
            x = Dropout(rate=0.25)(x)
        else:
          x = Activation('sigmoid')(x)
      decoder_output = x
      model = keras.Model(decoder_input, decoder_output)
      return model

    def call(self, inputs):
      '''
      '''
      return self.model(inputs)