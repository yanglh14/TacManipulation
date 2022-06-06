import numpy as np
import pickle

object_name = 'cube_small.pkl'

with open(object_name,'rb') as f:
    d = pickle.load(f)

d['class'] = np.ones(1000)*6

with open(object_name,'wb') as f:
    pickle.dump(d,f,pickle.HIGHEST_PROTOCOL)