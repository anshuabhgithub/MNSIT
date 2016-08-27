import numpy as nu
import pandas as pd
import dd
data =dd.data 
size = 41000
test_data = dd.test_data
x =data.iloc[:,1:].values
y_out  =data.iloc[:,0]
y_out = y_out.values
x[x>1] =1 
y = nu.zeros((x.shape[0],10))
for i in range(10):
	index = y_out == i
	y[index,i] =1


#x = x.values
#y = y.values

x_in = x[:size]
y_in = y[:size]
#x_in = x[:-5000]
#y_in = y[:-5000]
x_test  = x[-1000:]
y_test = y[-1000:]
