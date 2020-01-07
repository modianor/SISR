import keras.backend as K
import numpy as np
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.initializers import TruncatedNormal
from keras.layers import Input, Conv2D, concatenate, ELU
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from conf import myConfig as config

inint = TruncatedNormal(mean=0.0, stddev=0.05, seed=0)

inputs = Input(shape=(None, None, 1))

# CT去噪复现的模型
x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, name='block2_conv1', padding='same', kernel_initializer=inint)(
    inputs)
# x_1 =Activation('relu')(x)
x_1 = ELU(alpha=0.1)(x)

# conv_block1
x = Conv2D(filters=48, kernel_size=(3, 3), dilation_rate=(2, 2), name='block2_conv2_1', padding='same',
           kernel_initializer=inint)(x_1)
# x = BatchNormalization()(x)
x = ELU(alpha=0.1)(x)
# x=Activation('relu')(x)
x = Conv2D(filters=48, kernel_size=(3, 3), dilation_rate=(2, 2), name='block2_conv2_2', padding='same',
           kernel_initializer=inint)(x)
# x = BatchNormalization()(x)
x_2 = ELU(alpha=0.1)(x)
# x_2 =Activation('relu')(x)
x_3 = concatenate([x_2, x_1])

# conv_block2
x = Conv2D(filters=48, kernel_size=(3, 3), dilation_rate=(2, 2), name='block2_conv3_1', padding='same',
           kernel_initializer=inint)(x_3)
# x = BatchNormalization()(x)
x = ELU(alpha=0.1)(x)
# x =Activation('relu')(x)
x = Conv2D(filters=48, kernel_size=(3, 3), dilation_rate=(2, 2), name='block2_conv3_2', padding='same',
           kernel_initializer=inint)(x)
# x = BatchNormalization()(x)
x_4 = ELU(alpha=0.1)(x)
# x_4=Activation('relu')(x)
x_5 = concatenate([x_4, x_3])

# conv_block3
x = Conv2D(filters=48, kernel_size=(3, 3), dilation_rate=(3, 3), name='block2_conv4_1', padding='same',
           kernel_initializer=inint)(x_5)
# x = BatchNormalization()(x)
x = ELU(alpha=0.1)(x)
# x =Activation('relu')(x)
x = Conv2D(filters=48, kernel_size=(3, 3), dilation_rate=(3, 3), name='block2_conv4_2', padding='same',
           kernel_initializer=inint)(x)
# x = BatchNormalization()(x)
x_6 = ELU(alpha=0.1)(x)
# x_4 =Activation('relu')(x)
x_7 = concatenate([x_6, x_5])

# conv_block4
x = Conv2D(filters=48, kernel_size=(3, 3), dilation_rate=(2, 2), name='block2_conv5_1', padding='same',
           kernel_initializer=inint)(x_7)
# x = BatchNormalization()(x)
x = ELU(alpha=0.1)(x)
# x =Activation('relu')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=(2, 2), name='block2_conv5_2', padding='same',
           kernel_initializer=inint)(x)
# x = BatchNormalization()(x)
x_8 = ELU(alpha=0.1)(x)
# x_6 =Activation('relu')(x)
x_9 = concatenate([x_8, x_7])

# conv
x_10 = Conv2D(filters=1, kernel_size=(3, 3), strides=1, name='block2_conv5', padding='same',
              kernel_initializer=inint)(x_9)
model = Model(inputs=inputs, outputs=x_10)
model.summary()

# load the data and normalize it
cleanImages = np.load(config.data)
# print(cleanImages.dtype)

cleanImages = cleanImages / 255.0
cleanImages = cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")


def myFlow(generator, X):
    for batch in generator.flow(x=X, batch_size=config.batch_size, seed=7):
        trueNoiseBatch = np.random.normal(0.0, config.sigma / 255.0, batch.shape)
        noisyImagesBatch = batch + trueNoiseBatch
        yield (noisyImagesBatch, trueNoiseBatch)


# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha = 0.001
    factor = 0.5
    dropEvery = 5
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    return float(alpha)


callbacks = [LearningRateScheduler(lr_decay)]
# callbacks学习率调度器，该回调函数是用于动态设置学习率


# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt = optimizers.Adam(lr=0.01)


def custom_loss(y_true, y_pred):
    diff = y_true - y_pred
    res = K.sum(diff * diff) / (2 * config.batch_size)
    return res


model.compile(loss=custom_loss, optimizer=opt)

# train
model.fit_generator(myFlow(aug, cleanImages),
                    epochs=config.epochs, steps_per_epoch=len(cleanImages) // config.batch_size, callbacks=callbacks,
                    verbose=1)

# save the model
model.save('UNIQUE.h5')
