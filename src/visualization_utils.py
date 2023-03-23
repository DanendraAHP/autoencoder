import matplotlib.pyplot as plt
import random
import numpy as np

def visualize_sample(ds, ncol, sample_size):
    #variable
    img,label = next(iter(ds))
    if sample_size%ncol==0:
        nrow = sample_size//ncol
    else:
        nrow = sample_size//ncol+1
    #visualize
    fig, axs = plt.subplots(ncols=ncol,nrows=nrow, figsize=(12,10))
    for i in range(sample_size):
        if i%2==0:
            axs[i//ncol, i%ncol].imshow(img[i].numpy().astype("uint8"))
        else:
            axs[i//ncol, i%ncol].imshow(label[i].numpy().astype("uint8"))
        axs[i//ncol, i%ncol].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
def visualize_difference(ds, sample_size, autoencoder):
    #variable
    img,_ = next(iter(ds))
    encoded = autoencoder.encoder(img.numpy())
    decoded = autoencoder.decoder(encoded)
    samples = [i for i in range(img.shape[0])]
    sample_size = 5
    samples = random.choices(samples, k=sample_size)
    fig, axs = plt.subplots(ncols=2,nrows=5, figsize=(12,10))
    for i in range(sample_size):
        sample_idx = samples[i]
        axs[i,0].imshow(img[sample_idx].numpy().astype('uint8'))
        axs[i,0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i,1].imshow(decoded[sample_idx].numpy().astype('uint8'))
        axs[i,1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def interpolate(ds, autoencoder, interpolate_step):
    img,_ = next(iter(ds))
    encoded = autoencoder.encoder(img.numpy())
    img_1_idx = random.randint(0, img.shape[0])
    img_2_idx = random.randint(0, img.shape[0])
    while img_1_idx==img_2_idx:
        img_2_idx = random.randint(0, img.shape[0])
    encoded_img_1 = encoded[img_1_idx]
    encoded_img_2 = encoded[img_2_idx]
    ratios = np.linspace(0, 1, num=interpolate_step)
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * encoded_img_1 + ratio * encoded_img_2
        vectors.append(v)
    vectors = np.asarray(vectors)
    decoded = autoencoder.decoder(vectors)
    fig, axs = plt.subplots(nrows=1, ncols=interpolate_step, figsize=(30,5))
    for i in range(15):
        axs[i].imshow(decoded[i].numpy().astype('uint8'))
        axs[i].axis('off')