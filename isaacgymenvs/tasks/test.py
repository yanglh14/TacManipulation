import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

data = pd.read_csv('tactile_data',delim_whitespace=True)
data = data.values
tactile = data[:,26]
position = np.linalg.norm(data[:,2:5],axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.plot(-np.array(position-0.53219397) * 1000, label='position')
ax.plot(np.array(tactile)/100, label='force/100')

ax.legend()
plt.show()
