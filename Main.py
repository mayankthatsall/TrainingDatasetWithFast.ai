#This program will give you information about the GPU alloted to you
!nvidia-smi

from fastai.vision import *

#creating and managing the directories in my GDrive
!mkdir /root/.kaggle/
!cp drive/'My Drive'/kaggle.json /root/.kaggle/

#Copied API Command of required dataset from Kaggle
!kaggle datasets download -d mayankthatsall/my-helmets
!unzip ./my-helmets.zip

#Providing the paths
data_path=Path('./helmet').absolute()
model_path=Path('./model').absolute()

data=ImageDataBunch.from_folder(data_path,valid_pct=0.2,size=256,ds_tfms=get_transforms()).normalize(imagenet_stats)
print(data.classes, len(data.train_ds), len(data.valid_ds))
data.show_batch(rows=3)

#Loading our CNN
learn=cnn_learner(data,models.resnet50,metrics=error_rate,
                  model_dir=model_path)

#Training your model
learn.fit_one_cycle(4)
learn.unfreeze()

#Finding the learning rate and then plotting it's curve
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(6,max_lr=3e-4)

#Saving and exporting your trained model
learn.export(model_path / 'helmet.pkl')
learn.save(model_path/ 'helmet')
!cp model/helmet.pkl drive/'My Drive'/
