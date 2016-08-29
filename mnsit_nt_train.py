import imp
import con_layer as ly
import frw_out as fw
import numpy as nu
import matplotlib.pyplot as plt 
import pre_mndt as data

#copy assign training data set 
x = data.x_in
y = data.y_in
x_test = data.x_ts
y_test = data.y_ts 

#assign layer variable
ly1_nd = 100 
ly1_in = x.shape[1]
ly2_nd = 10 
ly2_in = ly1_nd

#create instances of layer
lay1 = ly.Nu_layer(ly1_nd,ly1_in)
#lay1.W = lay1.W*20
lay2 = ly.Nu_layer(ly2_nd,ly2_in)

#epoc size
mini_batch = 25 
ep_sz = 35 
data_size = x.shape[0]
alpha1 = .3
alpha2 = .1

#temp variable
pntr =0;
lay1_w = nu.zeros((ep_sz*data_size ,lay1.W.shape[0],lay1.W.shape[1]))
lay2_w = nu.zeros((ep_sz*data_size,lay2.W.shape[0],lay2.W.shape[1]))


#define forward pass
def frw_pass(xin):
	lay1.frw_pass(xin)
	lay2.frw_pass(lay1.out)


#define backward pass
def bck_pass(grad):
	lay2.bck_pass(grad)
	lay1.bck_pass(lay2.Grad_out)

#calculate initial cost
cost =0
cost_grad1 =0 
out = nu.empty_like(y)
#W_int = lay.W
cost =0
for i in range(x.shape[0]):
	frw_pass(x[i])
	cost=fw.cost(lay2.out,y[i]) +cost
print "intial cost is ", cost
cost =0
no_trance =data_size/mini_batch
for k in range(ep_sz):
	if(k>=0):
		alpha1 =.05
		alpha2 = .05
	index = nu.arange(data_size)
	nu.random.shuffle(index)
	x= x[index]
	y= y[index]
	no_trance =data_size/mini_batch
	data_in = nu.array_split(x,no_trance)
	data_out = nu.array_split(y,no_trance)
	for l in range(no_trance):
		x_temp = data_in[l]
		y_temp =data_out[l]
		for i in range(x_temp.shape[0]):
			frw_pass(x_temp[i])
			cost_grad1 = fw.intropy_cost_grad(y_temp[i],lay2.out)
			bck_pass(cost_grad1)

		#print "the weight is layer one is ", lay1.W
		lay1.update_weight(alpha1,x_temp.shape[0])
		lay2.update_weight(alpha2,x_temp.shape[0])
		lay1_w[pntr] = lay1.W
		lay2_w[pntr] = lay2.W
		pntr= pntr +1
		#print "the pntr is " ,pntr
	for m in range(x.shape[0]):
		frw_pass(x[m])
		cost=fw.cost(lay2.out,y[m]) +cost
	print "epoc ", k ," :train_cost ",cost
	cost =0
	for n in range(x_test.shape[0]):
		frw_pass(x_test[n])
		y_temp = nu.zeros_like(lay2.out)
		y_temp[nu.argmax(lay2.out)] = 1
		cost=fw.cost(y_temp,y_test[n]) +cost
	print "epoc ", k, " :test_cost ", cost
	cost = 0

for i in range(x.shape[0]):
	frw_pass(x[i])
	cost=fw.cost(lay2.out,y[i]) +cost
print "final cost is", cost
cost =0
def frw_out(xin):
	lay1.frw_pass(xin)
	lay2.frw_pass(lay1.out)
	return lay2.out
for i in range(x_test.shape[0]):
	frw_pass(x_test[i])
	y_temp = nu.zeros_like(lay2.out)
	y_temp[nu.argmax(lay2.out)] = 1
	#y_temp[y_temp<.5] = 0
	cost=fw.cost(y_temp,y_test[i]) +cost
print "test cost is", cost

i =range(data_size*ep_sz)
#plt.scatter(i,lay1_w[:,1,0],color ='red')
#plt.scatter(i,lay2_w[:,1,0],color ='blue')
