import numpy as nu
import pandas as pd
import dd
import mnist_loader as ld 
#data =dd.data 
size =49000
train_data,test_data,val_data = ld.load_data_wrapper()
#test_data = dd.test_data
train_data =nu.array(train_data)
test_data =nu.array(test_data)

x_train = train_data[:,0]
y_train = train_data[:,1]

x_test = test_data[:,0]
y_test = test_data[:,1]

train_size = train_data.shape[0]
test_size = test_data.shape[0]

print "test_size" ,train_size ,"data_size" ,x_test[1].shape[0]


x_in = nu.empty((train_size,x_train[1].shape[0]))
y_in = nu.empty((train_size,10))
y_ts = nu.zeros((test_size,10))
x_ts = nu.empty((test_size,x_test[1].shape[0]))
for i in range(train_size):
	x_in[i] = x_train[i].reshape(-1)
	y_in[i] = y_train[i].reshape(-1)


for i in range(test_size):
	x_ts[i] = x_test[i].reshape(-1)
	y_ts[i,y_test[i]] = 1;

#x_ts = x_ts[:-9000]
#y_ts = y_ts[:-9000]

x_in = x_in[:size]
y_in = y_in[:size]
