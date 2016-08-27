import numpy as nu
import math
def cost(y,out):
	return nu.sum((y-out)**2)

def cost_grad(y,out):
	return 2*(out-y)
def sigmoid(in_put):
	return 1/(1+nu.exp(-in_put))


def frw_out(W,in_put):
	return sigmoid(in_put.dot(W))
def intropy_cost_grad(y,out):
	return 2*(out-y)/((out*(1-out)))
