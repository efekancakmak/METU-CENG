import matplotlib.pyplot as plt
import numpy as np

colors = ['red', 'lime', 'fuchsia', 'dodgerblue', 'gold', 'darkcyan', 'darkgreen', 'mediumslateblue']

clusters11 = np.load('outs/hac_1_single.npy',allow_pickle=True)
clusters12 = np.load('outs/hac_1_complete.npy',allow_pickle=True)
clusters13 = np.load('outs/hac_1_average.npy',allow_pickle=True)
clusters14 = np.load('outs/hac_1_centroid.npy',allow_pickle=True)

clusters21 = np.load('outs/hac_2_single.npy',allow_pickle=True)
clusters22 = np.load('outs/hac_2_complete.npy',allow_pickle=True)
clusters23 = np.load('outs/hac_2_average.npy',allow_pickle=True)
clusters24 = np.load('outs/hac_2_centroid.npy',allow_pickle=True)

clusters31 = np.load('outs/hac_3_single.npy',allow_pickle=True)
clusters32 = np.load('outs/hac_3_complete.npy',allow_pickle=True)
clusters33 = np.load('outs/hac_3_average.npy',allow_pickle=True)
clusters34 = np.load('outs/hac_3_centroid.npy',allow_pickle=True)

clusters41 = np.load('outs/hac_4_single.npy',allow_pickle=True)
clusters42 = np.load('outs/hac_4_complete.npy',allow_pickle=True)
clusters43 = np.load('outs/hac_4_average.npy',allow_pickle=True)
clusters44 = np.load('outs/hac_4_centroid.npy',allow_pickle=True)

i=0
for c in clusters11:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters12:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters13:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters14:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()


i=0
for c in clusters21:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters22:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters23:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters24:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()


i=0
for c in clusters31:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters32:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters33:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters34:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()


i=0
for c in clusters41:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters42:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters43:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()
i=0
for c in clusters44:
    plt.scatter(c[:,0],c[:,1],color=colors[i])
    i+=1
plt.show()