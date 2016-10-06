
# coding: utf-8

# Introduction to Python Notebooks vs 1.3
# 
# // the idea of this script 'joe' is to introduce the basic metaphors for a novice to Python and Jupyter
# // it then goes on to explore some of the commonly used packages such as Pandas, Sci-Kit Learn and Numpy for data processing, statistical model building, comparison and  evaluation
# // we will explore some of the Visualisation packages such as Matplotlib and Seaborn
# // Finally we take a look at sharing and operationalising the value of Python
# 

# In[1]:

name='joe'


# In[2]:

name


# In[3]:

get_ipython().magic(u'matplotlib inline')


# In[5]:

import numpy as np
import matplotlib.pyplot as plt


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()


# """
# Shows how to combine Normalization and Colormap instances to draw
# "levels" in pcolor, pcolormesh and imshow type plots in a similar
# way to the levels keyword argument to contour/contourf.
# 
# """

# In[8]:

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np


# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(1, 5 + dy, dy),
                slice(1, 5 + dx, dx)]

z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, (ax0, ax1) = plt.subplots(nrows=2)

im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
ax0.set_title('Pcolormesh with levels')


# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                  y[:-1, :-1] + dy/2., z, levels=levels,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('Contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()


# In[10]:

get_ipython().run_cell_magic(u'timeit', u'', u'square_evens = [n*n for n in range(1000)]')


# In[11]:

get_ipython().run_cell_magic(u'HTML', u'', u'<iframe width="560" height="315" src="https://www.youtube.com/embed/cHZONQ2-x7I" frameborder="0" allowfullscreen></iframe>')


# In[3]:

import pandas as pd
import numpy as np


# In[7]:

df = pd.DataFrame(np.random.randn(10,5))
df.head()


# Introducing 'Iris' -  with some pretty flower graphs

# In[4]:

import pandas as pd
import seaborn as sns
sns.set(style="whitegrid", palette="muted")

# Load the example iris dataset
iris = sns.load_dataset("iris")

# "Melt" the dataset to "long-form" or "tidy" representation
iris = pd.melt(iris, "species", var_name="measurement")

# Draw a categorical scatterplot to show each observation
sns.swarmplot(x="measurement", y="value", hue="species", data=iris)


# In[ ]:



