import tensorflow as tf
from keras.models import Model
from config.config import TRAIN_CONFIG

class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      #tf.keras.layers.Rescaling(scale=1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
      tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu', input_shape=(TRAIN_CONFIG['IMG_HEIGHT'], 
                                                                                               TRAIN_CONFIG['IMG_WIDTH'],
                                                                                               TRAIN_CONFIG['NUM_CHANNEL'])),
      tf.keras.layers.MaxPooling2D(2, strides=2),
      tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(2, strides=2),
      tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(2, strides=2)
    ])
    

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu'),
      tf.keras.layers.UpSampling2D(2),
      tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', activation='relu'),
      tf.keras.layers.UpSampling2D(2),
      tf.keras.layers.Conv2D(3, 3, strides=1, padding='same', activation='relu'),
      tf.keras.layers.UpSampling2D(2),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def train(model, train_ds, val_ds):
  model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
  es = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss', mode='min', 
      verbose=1,
      patience=TRAIN_CONFIG['ES_PATIENCE']
  )  
  mc=tf.keras.callbacks.ModelCheckpoint(
      filepath = TRAIN_CONFIG['MODEL_PATH'], 
      monitor='val_loss', 
      mode='min', 
      save_best_only=True,
      verbose=1,
      save_weights_only=True
  )  
  history = model.fit(train_ds,
      epochs=TRAIN_CONFIG['EPOCHS'],
      shuffle=True,
      validation_data=val_ds,
      callbacks=[es, mc]
  )
  return history
