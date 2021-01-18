
# Optimizing Convolutional Neural Network Architecture via Information Field
**Stage-Level Organizer**

# Prerequisites
+ Python 3.6+
+ PyTorch 1.0+


## Training
+Avaiable Models [model name]: VGG16, VGG19, ResNet18, ResNet34, ResNet50, MobileNet
```
# Start training with: 
python main.py

# You can manually resume the training with: 

##Cifar-10
python main.py --resume --lr=0.01

##Imagenet
python main.py -a resnet18 --lr=0.1 [imagenet-folder with train and val folders]

```
