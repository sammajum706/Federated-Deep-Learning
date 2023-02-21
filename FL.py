import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
from sklearn.metrics import*
import math
from sklearn.preprocessing import label_binarize
import csv
import torchvision
from torchvision import datasets,models,transforms
import time
import os
import copy
import random
import matplotlib.pyplot as plt
import splitfolders
import shutil

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Splitting each of the client data into train and validation sets respectively
client=["Client1","Client2","Client3","Client4","Client5","Client6"]
for c in client:
    input_folder=os.getcwd()+"/"+c
    splitfolders.ratio(input_folder, output="New_"+c, seed=1337, ratio=(0.9,0.1), group_prefix=None)


def get_data(data_dir):
  
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])

    data_transforms={'train':transforms.Compose([transforms.Resize((224,224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean,std)]),
                   'val':transforms.Compose([transforms.Resize((224,224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean,std)])
                   }

  
    image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=10)
                 for x in ['train','val']}

    dataset_sizes={x:len(image_datasets[x]) for x in ['train','val']}
    class_names=image_datasets['train'].classes
    num_classes=len(class_names)
    return (image_datasets,dataloaders,dataset_sizes,class_names,num_classes)


#Function defined to train each of the client model
def model_train(model,lr,epochs,dataloaders,dataset_sizes,data_dir):
    criterion=nn.CrossEntropyLoss()
    since=time.time()
    best_acc=0.0
    optimizer=optim.SGD(model.parameters(),lr)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
    for epo in range(epochs):
        for phase in ['train','val']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss=0.0
            running_correct=0.0
            for inputs,labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                with torch.set_grad_enabled(phase=='train'):
                    outputs=model(inputs)
                    _,preds=torch.max(outputs,1)
                    loss=criterion(outputs,labels)
                    if phase=='train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                running_loss+=loss.item()*inputs.size(0)
                running_correct+=torch.sum(preds==labels.data)
            if phase == 'train':
                step_lr_scheduler.step()  
            epoch_loss=running_loss/dataset_sizes[phase]
            epoch_acc=running_correct/dataset_sizes[phase]
      
            if phase=='val' and epoch_acc>=best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())
          
    
    time_elapsed = time.time() - since
    print('Federated Client Model Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Federated Client Model Best val Acc: {:4f}'.format(best_acc))
    print()
    model.load_state_dict(best_model_wts)
    return model 


lr=0.01
epochs=10
model_name="Densenet121"


# Densenet 121 model pretrained on the imagenet dataset 
def Densenet121(n):
    model=models.densenet121(pretrained = True)
    num_ftrs = model.classifier.in_features
    model.classifier=nn.Linear(num_ftrs,n)
    model=model.to(device)
    return model

# Function defined to evaluate the test data
def test_result(data_dir,model_name,model,num_config):
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    Labels1=[]
    Predictions1=[]
    tr=transforms.Compose([transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean,std)])
    iden=datasets.ImageFolder(data_dir,tr)

    testloader=torch.utils.data.DataLoader(iden,batch_size=1,shuffle=False)
    model=model.eval()
    correct = 0
    num=0
    cd=["Benign","Malignant"]
    for i in range(len(cd)):
        for F in os.listdir(data_dir+"/"+cd[i]):
            Labels1.append(i)
    f = open(os.path.join(data_dir,model_name+".csv"),'w+',newline = '')
    writer = csv.writer(f)
    temp_array = np.zeros((len(testloader),len(iden.classes)))
    for inputs,labels in testloader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        with torch.set_grad_enabled(False):
            outputs=model(inputs)
            _,preds=torch.max(outputs,1)
            correct += (preds == labels.cuda()).sum().item()
            prob = torch.nn.functional.softmax(outputs, dim=1)
            temp_array[num] = np.asarray(prob[0].tolist()[0:len(iden.classes)])
            num+=1
    print(str(num_config+1)+". Federated Test Accuracy = ",100*correct/len(iden))
    Acc=correct/len(iden)
    for i in range(len(testloader)):
        writer.writerow(temp_array[i].tolist())
    f.close()
    for i in range(temp_array.shape[0]):
        mlx=np.max(temp_array[i])
        Predictions1.append(np.where(temp_array[i]==mlx)[0][0])
    matrix = confusion_matrix(Labels1, Predictions1)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    return Acc


def val_result(datadir,model_name,model,num_config,c):
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    Labels=[]
    Predictions=[]
    tr=transforms.Compose([transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean,std)])
    iden=datasets.ImageFolder(datadir,tr)

    valloader=torch.utils.data.DataLoader(iden,batch_size=1,shuffle=False)
    model=model.eval()
    correct = 0
    num=0
    for i in range(len(os.listdir(datadir))):
        for f in os.listdir(datadir+"/"+os.listdir(datadir)[i]):
            Labels.append(i)
    temp_array = np.zeros((len(valloader),len(iden.classes)))    
    for inputs,labels in valloader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        with torch.set_grad_enabled(False):
            outputs=model(inputs)
            _,preds=torch.max(outputs,1)
            correct += (preds == labels.cuda()).sum().item()
            prob = torch.nn.functional.softmax(outputs, dim=1)
            temp_array[num] = np.asarray(prob[0].tolist()[0:len(iden.classes)])
            num+=1

    for i in range(temp_array.shape[0]):
        mlx=np.max(temp_array[i])
        Predictions.append(np.where(temp_array[i]==mlx)[0][0])
    matrix = confusion_matrix(Labels, Predictions)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("Entire Result before "+str(num_config+1)+" for "+c+": ",100*correct/len(iden))

# Function which is defined for taking out the average of the weights of the client model 
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] +=w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


client=["New_Client1","New_Client2","New_Client3","New_Client4","New_Client5","New_Client6"]
old_client=["Client1","Client2","Client3","Client4","Client5","Client6"]
num_config=10
j=0


# The central model and the client models are being created
central_model=Densenet121(2)
model_1=Densenet121(2)
model_2=Densenet121(2)
model_3=Densenet121(2)
model_4=Densenet121(2)
model_5=Densenet121(2)
model_6=Densenet121(2)


model_list=[model_1,model_2,model_3,model_4,model_5,model_6]


test_dir="/test"
Server_Accuracy=[]

for i in range(num_config):
    print('Communication {}/{}'.format(i+1,num_config))
    print('-'*10)
    Weights=[]
    j=0
    if i==0:
        best_model_wts=central_model.state_dict()
    
    for phase in ['train','test']:
        if phase=='train':
            for m in model_list:
                data_dir="/FL/"+client[j]
                os.chdir(data_dir)
                val_result("/FL/"+old_client[j],model_name,central_model,i,old_client[j])
                image_datasets,dataloaders,dataset_sizes,class_names,num_classes=get_data(data_dir)
                m.load_state_dict(best_model_wts)
                m=model_train(m,lr,epochs,dataloaders,dataset_sizes,data_dir)
                w=m.state_dict()
                Weights.append(copy.deepcopy(w))
                j+=1  
            best_model_wts=average_weights(Weights)
            central_model.load_state_dict(best_model_wts)
             
        else:   
            acc=test_result(test_dir,str(i)+"_"+model_name,central_model,i)
            Server_Accuracy.append(acc)
        print()



# Server Accuracy Plot
plt.title("{} after communication: {}".format("Accuracy",len(Server_Accuracy)))
plt.xlabel("Communication")
plt.ylabel("Accuracy")
plt.plot(list(range(len(Server_Accuracy))),Server_Accuracy,color="g")
os.chdir("/FL")
plt.savefig("Server_Accuracy.png")
plt.close()