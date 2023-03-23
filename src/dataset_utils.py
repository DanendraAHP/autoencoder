import tensorflow as tf
from config.config import TRAIN_CONFIG
AUTOTUNE = tf.data.AUTOTUNE

class Dataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.val_frac = TRAIN_CONFIG['VALIDATION_SIZE']
        self.rescale = TRAIN_CONFIG['RESCALE']
        self.noise_factor = TRAIN_CONFIG['NOISE_FACTOR']
        self.use_noise = TRAIN_CONFIG['USE_NOISE']
    def get_img(self, filepath):
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img, channels=TRAIN_CONFIG['NUM_CHANNEL'])
        img = tf.image.resize(img, [TRAIN_CONFIG['IMG_HEIGHT'], TRAIN_CONFIG['IMG_WIDTH']])
        if self.rescale:
            img = tf.keras.layers.Rescaling(scale=TRAIN_CONFIG['SCALE'])(img)
        return img, img
    def add_noise(self, img, label):
        if self.rescale:
            img_noisy = img+self.noise_factor*tf.random.normal(shape=img.shape)
            img_noisy = tf.clip_by_value(img_noisy, clip_value_min=0.,clip_value_max=1.)
        else:
            img_noisy = img+self.noise_factor*tf.random.normal(shape=img.shape)*255
            img_noisy = tf.clip_by_value(img_noisy, clip_value_min=0.,clip_value_max=255.)
        return img_noisy, label
    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(TRAIN_CONFIG['BATCH_SIZE'])
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    def prepare_train_data(self):
        ds = tf.data.Dataset.list_files(f'{self.data_dir}/*/*.jpeg')
        total_img = ds.cardinality().numpy()
        val_size = int(self.val_frac*total_img)
        #split to train and val
        self.train_ds = ds.skip(val_size)
        self.val_ds = ds.take(val_size)
        #preprocess the data
        self.train_ds = self.train_ds.map(self.get_img, num_parallel_calls=AUTOTUNE)
        self.val_ds = self.val_ds.map(self.get_img, num_parallel_calls=AUTOTUNE)
        #if want to add noise to the data
        if self.use_noise:
            self.train_ds = self.train_ds.map(self.add_noise)
            self.val_ds = self.val_ds.map(self.add_noise)
        #optimize the data
        self.train_ds = self.configure_for_performance(self.train_ds)
        self.val_ds = self.configure_for_performance(self.val_ds)