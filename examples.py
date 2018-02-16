#python wrapper for cut pursuit
import numpy as np
import libCP
#TOY EXAMPLE
#
#  V1 -- V2 -- V3
#
# F(x) = 1/2 * (4 *(x1-10)^2 + 4 * (x2)^2 + 4 * (x3-10)^2)
#      + 3 * |x1-x2| + 3 * |x2-x3| 
#      + 0.5* (|x1| + |x2| + |x3|)
source = np.array([0, 1], dtype = 'uint32')
target = np.array([1, 2], dtype = 'uint32')
obs = np.array([10.,0.,10.], dtype = 'float32')
edge_weight = 3*np.array([1., 1.], dtype = 'float32')
A = np.array([[2, 0., 0,], [0., 2., 0,], [0., 0., 2,]], dtype = 'float32')
l1_weight = 0.5 * np.array([1., 1., 1.], dtype = 'float32')

#all the following calls give the same results:
cV, rX = libCP.CP_PFDR_graph(obs, source, target, edge_weight, A, l1_weight, 0, 2)
cV, rX = libCP.CP_PFDR_graph(obs, source, target, edge_weight, A, 0.5, 0, 2)
cV, rX = libCP.CP_PFDR_graph(obs, source, target, 3, A, 0.5, 0, 2)
cV, rX = libCP.CP_PFDR_graph(obs, source, target, 3, np.array([2, 2., 2,], dtype = 'float32'), 0.5, 0, 2)
cV, rX = libCP.CP_PFDR_graph(obs, source, target, 3, 2, 0.5, 0, 2)

#EEG example
#see ``Cut-Pursuit Algorithm for Regularizing Nonsmooth Functionals with 
#Graph Total Variation`` for details
#Courtesy of Ahmad Karfoul et Isabelle Merlet, LTSI,INSERM U1099.
import scipy.io
from timeit import default_timer as timer
eeg = scipy.io.loadmat('data_CP_EEG.mat')

obs = np.array(eeg["y"][:,0], dtype='f8')
source = np.array(eeg["Eu"][:,0], dtype='uint32')
target = np.array(eeg["Ev"][:,0], dtype='uint32')
edge_weight = np.array(eeg["La_d1"][:,0], dtype='f8')
A = np.array(eeg["Phi"], dtype='f8')
l1_weight = np.array(eeg["La_l1"][:,0], dtype='f8')

start = timer()
cV, rX = libCP.CP_PFDR_graph(obs, source, target, edge_weight, A, l1_weight, 0, 0)
print("slow setting = %f" % (timer()-start))
start = timer()
cV, rX = libCP.CP_PFDR_graph(obs, source, target, edge_weight, A, l1_weight, 0, 1)
print("standard setting = %f" % (timer()-start))
start = timer()
cV, rX = libCP.CP_PFDR_graph(obs, source, target, edge_weight, A, l1_weight, 0, 2)
print("fast setting = %f" % (timer()-start))