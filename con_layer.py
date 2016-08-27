import random
import numpy as np
import pandas as pd
import frw_out as fp
class Nu_layer:

	def __init__(self,node,in_put):
		self.num_node = node
		self.num_input =in_put
		#self.num_input =1
		#self.W = np.random.random((node,in_put))/900
		self.W = np.random.randn(node,in_put)/in_put
		self.bias =  np.random.random(node)
		#self.W = np.r_[0.5];
		#self.Grad = nu.zeros((num_node,num_imput))
		self.Grad = np.zeros_like(self.W)
		self.b_grad = np.zeros_like(self.bias)

	def update_weight(self,alpha,N):
		self.W = self.W - alpha*self.Grad/N
		self.bias = self.bias - alpha*self.b_grad/N
		self.Grad = np.zeros_like(self.W)
		self.b_grad = np.zeros_like(self.bias)
		#print "after updation  " 
		#self.Grad
	def frw_pass(self,in_put):
		self.in_put =in_put
		#in_put = in_put[:,np.newaxis]
		self.out = fp.sigmoid(self.W.dot(in_put)+self.bias)

	def bck_pass(self,grad_in):
		#out = self.out
		#out  = out[:,np.newaxis]
		temp_grad = self.out*(1-self.out)*grad_in
		self.b_grad = self.b_grad +temp_grad
		temp_grad = temp_grad[:,np.newaxis]
		self.Grad_temp=self.in_put*temp_grad
		self.Grad_out = self.W.T.dot((self.out*(1-self.out))*grad_in)
		#self.Grad_out = self.Grad_temp*self.W
		#self.Grad_out= np.sum(self.Grad_out,0)
		self.Grad = self.Grad + self.Grad_temp
	def frw_out(self,in_put):
		self.frw_pass(in_put)
		return self.out




