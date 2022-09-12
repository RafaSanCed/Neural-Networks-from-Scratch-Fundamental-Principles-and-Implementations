# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:36:35 2022

@author: rafae
"""

import mnist_loader
import network
import pickle
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
net=network.Network([784,30,30,10])
net.SGD( training_data, 100, 10, 0.5, test_data=test_data)
archivo = open("red_prueba1.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()