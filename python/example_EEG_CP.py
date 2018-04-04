            #----------------------------------------------------%
            #  script for illustrating CP on EEG synthetic data  %
            #----------------------------------------------------%
""" dataset courtesy of Ahmad Karfoul and Isabelle Merlet, LTSI, INSERM U1099.
 
 Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
 Nonsmooth Functionals with Graph Total Variation.

 Loic Landrieu 2018"""
import numpy as np
import libCP
import scipy.io

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

#load data
eeg = scipy.io.loadmat('../data/EEG.mat')
vertices = eeg["mesh"]["v"][0][0]
faces = eeg["mesh"]["f"][0][0].astype('int')-1
activity_true=np.array(eeg["x0"][:,0], dtype='f8')
x0=np.array(eeg["x0"][:,0], dtype='f8')
obs = np.array(eeg["y"][:,0], dtype='f8')
source = np.array(eeg["Eu"][:,0], dtype='uint32')
target = np.array(eeg["Ev"][:,0], dtype='uint32')
edge_weight = np.array(eeg["La_d1"][:,0], dtype='f8')
A = np.array(eeg["Phi"], dtype='f8')
l1_weight = np.array(eeg["La_l1"][:,0], dtype='f8')

#function to represent the brain activity
def plot_brain(activity,title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(30, 90)
    cmap = plt.get_cmap('hot')
    collec = ax.plot_trisurf(vertices[:,0],vertices[:,1], vertices[:,2],triangles=faces,
                            cmap=cmap, antialiased=True, vmin=0, vmax=10, linewidth=0.1, edgecolor='Gray')
    collec.set_array(activity)
    plt.axis('off')
    plt.title(title)
    plt.show()

#solve the inverse problem
cV, rX = libCP.CP_quadratic_l1(obs, source, target, edge_weight, A, l1_weight, positivity=1, PFDR_rho=1.5)
activity_estimated= rX[cV]

#compute the support of the solution and our estimate
support_true = activity_true!=0
support_estimated = activity_estimated!=0

#compute a clean support using a heuristic based on 2-means
abs_x = abs(activity_estimated)
sabs_x = np.sort(abs_x)
n_0 = 0; n_1 = len(abs_x); # size of the two clusters
sum_0 = 0; sum_1 = sum(abs_x); # sum of the elements of the two clusters
m = sum_1/n_1
while (2*sabs_x[n_0] < m):
    n_0 = n_0 + 1
    n_1 = n_1 - 1
    sum_0 = sum_0 + sabs_x[n_0]
    sum_1 = sum_1 - sabs_x[n_0]
    m = (sum_0/n_0 + sum_1/n_1)
support_clean= abs_x > (m/2)

#compute the dice scores
DS_estimated = 2*sum(support_true * support_estimated)/(sum(support_true) + sum(support_estimated))
DS_clean = 2*sum(support_true * support_clean)/(sum(support_true) + sum(support_clean))

print("Dice score for raw retrieved activity:  %.2f; processed support: %.2f\n" % (DS_estimated, DS_clean));

#plot the activity
plot_brain(activity_true, 'ground truth ')
plot_brain(activity_estimated, 'raw retrieved activity ')
plot_brain(10*support_clean, 'identified sources ')
