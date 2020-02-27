# Imports
from keras.layers import Activation,Dense,Conv2D,UpSampling2D,LeakyReLU, Reshape, Flatten, Input,BatchNormalization,Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import numpy as np


def build_discriminator(img_shape):
    model = Sequential() #64x64 original shape
    
    model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=img_shape, padding="same")) #32x32
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=5, strides=2, padding="same")) #16x16
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=5, strides=2, padding="same")) #8x8
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten()) 
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(input=img, output=d_pred)

def build_generator(z_dimension, channels):
    model = Sequential()

    model.add(Dense(128 * 16 * 16, input_dim=z_dimension))
    model.add(LeakyReLU(alpha=0.2)) 
    model.add(Reshape((16, 16, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=5, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=5, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(channels, kernel_size=5, padding="same"))
    model.add(Activation("tanh"))

    model.summary()
    noise = Input(shape=(z_dimension,))
    img = model(noise)
    return Model(input=noise, output=img)

def sample_images(epoch):
    r, c = 4, 5
    noise = np.random.normal(0, 1, (r * c,z_dimension))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(PATH + "output64_k5_z64/%d.png" % epoch, dpi=200)
    plt.close()

#load real pictures:
with open("cat_dataset_64x64.pickle", "rb") as file:
    x_train = pickle.load(file)
x_train = x_train.reshape(-1,64,64,1)       
x_train = x_train / 127.5 - 1.  # values -1 to 1

# model parameters
PATH = "d:/ML/8j_CAT-GAN/"
img_rows = 64
img_cols = 64
channels = 1
img_shape = (img_rows, img_cols, channels)
z_dimension = 64
optimizer = Adam(0.0005, 0.5)

# build discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

# bild generator
generator = build_generator(z_dimension,channels)

# the generator takes noise as input and generates imgs
z = Input(shape=(z_dimension,))
img = generator(z)
discriminator.trainable = False
d_pred = discriminator(img)
# The combined model  (stacked generator and discriminator)
combined = Model(z, d_pred)
combined.compile(loss='binary_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])


# training parameters
epochs = 30000
batch_size = 64
sample_interval=1000 # save some generated pictrures

# adversarial ground truths
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

#training
for epoch in range(epochs):
    # real images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs = x_train[idx]
    # generated images
    noise = np.random.normal(0, 1, (batch_size, z_dimension))
    gen_imgs = generator.predict(noise)
    # train discriminator
    d_loss_real = discriminator.train_on_batch(imgs, real)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # train generator
    noise = np.random.normal(0, 1, (batch_size, z_dimension))
    g_loss = combined.train_on_batch(noise, real)
    # save progress
    if (epoch % sample_interval) == 0:
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % 
            (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
        sample_images(epoch)
        generator.save("generator_64_64_z64_%d_epoch.h5" % epoch)
