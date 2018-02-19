#python wrapper for cut pursuit
import numpy as np
import libCP
#TOY EXAMPLE
#
#  V1 -- V2 -- V3
#
# F(x) = 1/2 * ((x1-10)^2 + (x2)^2 + (x3-10)^2)
#      + 3 * |x1-x2| + 3 * |x2-x3| 
#      + 0.5* (|x1| + |x2| + |x3|)
data_type = 'f4'
source = np.array([0, 1], dtype='uint32')
target = np.array([1, 2], dtype='uint32')
obs = np.array([10.,0.,10.], dtype=data_type)
edge_weight = 3*np.array([1., 1.], dtype=data_type)
A = np.array([[1, 0., 0,], [0., 1., 0,], [0., 0., 1,]],dtype =data_type)
l1_weight = 0.5 * np.array([1., 1., 1.], dtype=data_type)
positivity = 0

#all the following calls give the same results:
cV, rX = libCP.CP_quadratic_l1(obs, source, target, edge_weight, A, l1_weight)
print(rX[cV])
cV, rX = libCP.CP_quadratic_l1(obs, source, target, edge_weight, A, 0.5)
print(rX[cV])
cV, rX = libCP.CP_quadratic_l1(obs, source, target, 3., A, 0.5)
print(rX[cV])
cV, rX = libCP.CP_quadratic_l1(obs, source, target, 3., np.array([1., 1., 1.,], dtype =data_type), 0.5)
print(rX[cV])
cV, rX = libCP.CP_quadratic_l1(obs, source, target, 3., 1., 0.5)
print(rX[cV])

#EEG example
#see ``Cut-Pursuit Algorithm for Regularizing Nonsmooth Functionals with 
#Graph Total Variation`` for details
#Courtesy of Ahmad Karfoul et Isabelle Merlet, LTSI,INSERM U1099.
import scipy.io
eeg = scipy.io.loadmat('data_CP_EEG.mat')

obs = np.array(eeg["y"][:,0], dtype='f8')
source = np.array(eeg["Eu"][:,0], dtype='uint32')
target = np.array(eeg["Ev"][:,0], dtype='uint32')
edge_weight = np.array(eeg["La_d1"][:,0], dtype='f8')
A = np.array(eeg["Phi"], dtype='f8')
l1_weight = np.array(eeg["La_l1"][:,0], dtype='f8')

cV, rX = libCP.CP_quadratic_l1(obs, source, target, edge_weight, A, l1_weight, positivity=1)
