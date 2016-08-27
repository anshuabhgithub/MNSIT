import numpy as nu
import mnsit_nt_train as tr
import frw_out as frw

ts_in  = tr.x_test
ts_out  = tr.y_test
fun = tr.frw_out
y_actual = nu.zeros_like(ts_out)
for i in range(ts_in.shape[0]):
	y_actual[i] = fun(ts_in[i])


y_actual[y_actual>=.5] =1 
y_actual[y_actual<.5] =0 

cost  = frw.cost(y_actual,ts_out)

