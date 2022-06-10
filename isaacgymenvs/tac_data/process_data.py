import numpy as np
import pickle
save_dir = ''
object_name = '010_potted_meat_can_dynamic'
object_name_2 = '011_banana_dynamic'
object_name_3 = '025_mug_dynamic'
object_name_4 = '061_foam_brick_dynamic'
object_name_5 = 'ball_small_dynamic'
object_name_6 = 'ball_big_dynamic'
object_name_7 = 'cube_small_dynamic'
object_name_8 = 'cube_big_dynamic'
object_name_9 = 'cylinder_small_dynamic'
object_name_10 = 'cylinder_big_dynamic'

with open(save_dir+object_name+'.pkl','rb') as f:
    d_1 = pickle.load(f)
with open(save_dir+object_name_2+'.pkl','rb') as f:
    d_2 = pickle.load(f)
with open(save_dir+object_name_3+'.pkl','rb') as f:
    d_3 = pickle.load(f)
with open(save_dir+object_name_4+'.pkl','rb') as f:
    d_4 = pickle.load(f)
with open(save_dir+object_name_5+'.pkl','rb') as f:
    d_5 = pickle.load(f)
with open(save_dir+object_name_6+'.pkl','rb') as f:
    d_6 = pickle.load(f)
with open(save_dir+object_name_7+'.pkl','rb') as f:
    d_7 = pickle.load(f)
with open(save_dir+object_name_8+'.pkl','rb') as f:
    d_8 = pickle.load(f)
with open(save_dir+object_name_9+'.pkl','rb') as f:
    d_9 = pickle.load(f)
with open(save_dir+object_name_10+'.pkl','rb') as f:
    d_10 = pickle.load(f)
d = {}
for item in d_1:
    d[item] = np.concatenate([d_1[item],d_2[item],d_3[item],d_4[item],d_5[item],d_6[item],d_7[item],d_8[item],d_9[item],d_10[item]],axis=0)

with open('tac_data_dynamic.pkl', 'wb') as f:
    pickle.dump(d,f,pickle.HIGHEST_PROTOCOL)