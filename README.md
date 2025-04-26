
# Data-Inherent Vulnerabilities: A Novel Framework for Model-Agnostic Backdoor Attacks

## BEFORE YOU RUN OUR CODE

We appreciate your interest in our work and trying out our code.If you have any  questions,please feel free to open an issue or contact any of the authors directly.We are more than happy to help you debug your experiment and find out the correct configuration.图链接+代码链接

## ABOUT
This repository contains code implementation of the paper "Data-Inherent Vulnerabilities: A Novel Framework for Model-Agnostic Backdoor Attacks"


## DEPENDENCIES

Our code is implemented and tested on pytorch. Following packages are used by our code.
```
torch==2.2.2
```
```
torchvision==0.17.2
```
```
numpy==1.26.4
```
```
nvidia-cublas-cu12==12.1.3.1
```
```
pandas==2.2.3
```




## HOW TO

### Train model

Before poisoning, it is necessary to pre train the model to prepare for subsequent fine-tuning. The train model code is under the AAAA. You will need to download the training data according to the instructions in the code. Please modify the code to the dataset and model you want to train first. 
Please run
```python
python train_model.py
```


### Inject backdoor and Finetune

**This is the main code.** Firstly, identify vulnerability points based on mathematical statistical characteristics, modify the pixel values of these vulnerability points, and finally fine tune the initial model obtained in the previous step. Finally,calculate ASR and ACC. Please note that these processes are all automated and only require modifications to the dataset and model. 
In order to complete this step,please run 
```python
python main.py
```


### Data Augmentation

Train a data augmentation model to observe any changes in attack success rate,please run 
```python
python train_model_data-aug.py
```


