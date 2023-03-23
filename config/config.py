TRAIN_CONFIG = {
    'IMG_WIDTH' : 200,
    'IMG_HEIGHT' : 200,
    'NUM_CHANNEL' :3,
    'BATCH_SIZE' : 32,
    'VALIDATION_SIZE' : 0.2,#must be 0-1,
    'MODEL_PATH' : 'model/cnn_ae_noisy/cp.ckpt',
    'EPOCHS' : 100,
    'ES_PATIENCE' : 5,
    'RESCALE' : False,
    'SCALE' : 1./255,
    'NOISE_FACTOR':0.2,
    'USE_NOISE' :True
}