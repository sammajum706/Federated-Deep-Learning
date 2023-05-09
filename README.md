# Federated-Deep-Learning

This repository is the federated deep learning implementation on the 40X magnification of BreakHis dataset using Densenet 121 model.

Here lets know about what is Federated Deep Learning technique in a brief manner :

* Federated learning aims at training a machine learning algorithm, for instance deep neural networks, on multiple local datasets contained in local nodes without explicitly exchanging data samples (local datasets are being treated in the form of clients). 

* The general principle consists in training local models on client data samples and exchanging parameters (e.g. the weights and biases of a deep neural network) between these local nodes at some frequency to generate a global model shared by all nodes.

* For the creation of the global model averaging of the weights of all the server models is being performed.

* The optimisation of the federated learning model is being done with the help of communication rounds. 


<br>
Manupulations done with the 40X maginification dataset :

* Here the 40X magnification images of the [BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis) dataset is used and the 1995 images is being splitted to training and test sets of 1395 and 600 images respectively.

* The 1395 training images is augmented to 3920 images and is being divided into six clients.

<br>
Here the client data distribution is shown below.<br><br>
<img src="/Dataset Distribution.png" style="margin: 10px;">


<br>
The overall working of the Federated Learning approach over the client datasets is being presented :
<br><br>
<img src="/Flow Chart.png" style="margin: 10px;">

