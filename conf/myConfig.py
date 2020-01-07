#import packages
import os

#default number of epochs
epochs=5

#default batch size for training
batch_size=64

#noise-level for training
sigma=0.0 #change it according to noise level in your dataset
# sigma = 0.0
# alpha=0.1
# beta=0.2


#path to generate the data
genDataPath='./data/'

#path to save the genPatches
save_dir='./training/'

#path to training data
data='./training/img_clean_pats.npy'

#variables required to generate patch,分割之后图片的大小
pat_size=40
stride=10
step=0




####### speckle噪声：
# noisyImageBatch = batch + config.alpha * np.log(
#             np.random.rayleigh(0.7, batch.shape) + config.beta) \
#                      + np.random.normal(0.0, 0.1, batch.shape)