
from keras.models import load_model
from conf import myConfig as config
import cv2
import numpy as np
from skimage.measure import compare_psnr
import argparse
from pathlib import Path
import keras.backend as K

#ParsingArguments
parser=argparse.ArgumentParser()
parser.add_argument('--dataPath',dest='dataPath',type=str,default='./Set12/',help='testDataPath')
parser.add_argument('--weightsPath',dest='weightsPath',type=str,default='./UNIQUE.h5')
args=parser.parse_args()

#createModel, loadWeights
def custom_loss(y_true,y_pred): #this is required for loading a keras-model created with custom-loss
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res

nmodel=load_model(args.weightsPath, custom_objects={'custom_loss':custom_loss})
print('nmodel is loaded')
# nmodel.summary()

#createArrayOfTestImages
p=Path(args.dataPath)
listPaths=list(p.glob('./*.png'))
imgTestArray = []
for path in listPaths:
    imgTestArray.append(((cv2.resize
    # (cv2.imread(str(path),0),(200,200),  #v1
    # (cv2.imread(str(path), 0), (180, 180), #v2
    # (cv2.imread(str(path), 0), (741, 512), #ELU-CNN
    # (cv2.imread(str(path), 0), (243, 504),  #MyI1.jpg
    # (cv2.imread(str(path), 0), (652, 482), #MyI1.jpg
    # (cv2.imread(str(path), 0), (519, 446),  # MyI0.jpg
    (cv2.imread(str(path), 0), (243, 507),  # SR.png
      interpolation=cv2.INTER_CUBIC))))
imgTestArray=np.array(imgTestArray)/255

# calculatePSNR
sumPSNR=0
for i in range(0,len(imgTestArray)):
    cv2.namedWindow('trueCleanImage', cv2.WINDOW_NORMAL)
    cv2.imshow('trueCleanImage',imgTestArray[i])
    cv2.waitKey(0)
    noisyImage=imgTestArray[i]+np.random.normal(0.0,config.sigma/255,imgTestArray[i].shape)
    cv2.namedWindow('noisyImage',cv2.WINDOW_NORMAL)
    cv2.imshow('noisyImage',noisyImage)
    cv2.waitKey(0)
    #print(np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0).shape)
    error=nmodel.predict(np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0))
    # cv2.imshow('Error', np.squeeze(error)); cv2.waitKey(0)
    predClean=noisyImage-np.squeeze(error)
        #print(error.min(),error.max())
    cv2.namedWindow('predCleanImage', cv2.WINDOW_NORMAL)
    cv2.imshow('predCleanImage',predClean)
    cv2.waitKey(0)
    psnr=compare_psnr(imgTestArray[i],predClean)
    sumPSNR=sumPSNR+psnr
    cv2.destroyAllWindows()
avgPSNR=sumPSNR/len(imgTestArray)
print('avgPSNR on test-data',avgPSNR)
print(sumPSNR)


# CleanImage=cv2.imread(r'C:\Users\86157\PycharmProjects\untitled\ELU-CNN\01.png',0)
# cv2.imshow('trueCleanImage',CleanImage/255)
# cv2.waitKey(0)
# noisyImage=CleanImage/255+np.random.normal(0.0,config.sigma/255,CleanImage.shape)
# cv2.imshow('noisyImage',noisyImage)
# cv2.waitKey(0)
# #print(np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0).shape)
# error=nmodel.predict(np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0))
# cv2.imshow('Error', np.squeeze(error))
# cv2.waitKey(0)
# predClean=noisyImage-np.squeeze(error)
#     #print(error.min(),error.max())
# cv2.imshow('predCleanImage',predClean)
# cv2.waitKey(0)
# psnr=compare_psnr(CleanImage,predClean)
# cv2.destroyAllWindows()
# print(psnr)
# import os
#
# path=os.getcwd()
# gan_M = load_model(path+r'\models\gan.h5')
# CleanImage=cv2.imread(r'C:\Users\86157\PycharmProjects\untitled\ELU-CNN\MyI9.jpg',0)
#
# cv2.imshow('cleanImage',CleanImage)
# cv2.waitKey(0)
# # cv2.imshow('noisyImage',ww/255)
# # # cv2.waitKey(0)
# noisyImage=CleanImage/255+np.random.normal(0.0,config.sigma/255,CleanImage.shape)
# cv2.imshow('noisyImage',noisyImage)
# cv2.waitKey(0)
# pre_noisy=(gan_M.predict((noisyImage).reshape(1,512,741,1))).reshape(512,741)
# # ss=(generator_M.predict((ww/255).reshape(1,512,741,1))).reshape(512,741)
# cv2.imshow('pred_noisyImage',pre_noisy)
# cv2.waitKey(0)
# pre_clean=noisyImage-pre_noisy
# cv2.imshow('pre_cleanimage',pre_clean)
# cv2.waitKey(0)
#
# # ss=noise_imgs[0]-gan.layers[1].predict(noise_imgs[0].reshape(-1,40,40,1))
# # ww=cv2.imread('MyI9.jpg', 0)
# # ss=(ww.reshape(1,512,741,1)-gan.layers[1].predict(ww.reshape(1,512,741,1))).reshape(512,741)
# # cv2.imshow('noisyImage',ww)
# # cv2.waitKey(0)
# # cv2.imshow('predCleanImage',ss.reshape(512, 741))
# # cv2.waitKey(0)