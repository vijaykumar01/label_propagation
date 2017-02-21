import numpy as np

with np.load('pipa_feas.npz') as data:
     feas = data['arr_0']

np.savetxt('pipa_feas.txt',feas);

with np.load('pipa_labels.npz') as data:
     feas = data['arr_0']

np.savetxt('pipa_labels.txt',feas);
