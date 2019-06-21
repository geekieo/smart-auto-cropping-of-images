
# coding: utf-8

# In[1]:

#get_ipython().magic(u'reload_ext autoreload')
#get_ipython().magic(u'autoreload 2')
#get_ipython().magic(u'matplotlib inline')
import sys
sys.path.append("D:\\PySpace\\SACI")

from salicon.salicon import SALICON
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[2]:

dataDir='test_data'
dataType='train2014examples'
annFile='%s/annotations/fixations_%s.json'%(dataDir,dataType)


# In[3]:

# initialize COCO api for instance annotations
salicon=SALICON(annFile)


# In[4]:

# get all images 
imgIds = salicon.getImgIds();
img = salicon.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]


# In[6]:

# load and display image
I = io.imread('%s/images/%s'%(dataDir,img['file_name']))
plt.figure()
plt.subplot(131)
plt.imshow(I)



# In[7]:

# load and display instance annotations
annIds = salicon.getAnnIds(imgIds=img['id'])
anns = salicon.loadAnns(annIds)
plt.subplot(132)
salicon.showAnns(anns)



# In[8]:

# show fixations
sal_map = salicon.buildFixMap(anns, doBlur=False)
print(np.min(sal_map), np.max(sal_map)) # 0.0 1.0
plt.subplot(133)
plt.imshow(sal_map, cmap = cm.Greys_r,vmin=0,vmax=1)
plt.show()


# In[ ]:



