import numpy as np
import hac

colors = ['red', 'lime', 'fuchsia', 'dodgerblue', 'gold', 'darkcyan', 'darkgreen', 'mediumslateblue']

#def hac(data, criterion, stop_length):

set1 = np.load("hac/dataset1.npy")
set2 = np.load("hac/dataset2.npy")
set3 = np.load("hac/dataset3.npy")
set4 = np.load("hac/dataset4.npy")

np.save("outs/hac_1_single.npy",hac.hac(set1,hac.single_linkage,2))
np.save("outs/hac_2_single.npy",hac.hac(set2,hac.single_linkage,2))
np.save("outs/hac_3_single.npy",hac.hac(set3,hac.single_linkage,2))
np.save("outs/hac_4_single.npy",hac.hac(set4,hac.single_linkage,4))

np.save("outs/hac_1_complete.npy",hac.hac(set1,hac.complete_linkage,2))
np.save("outs/hac_2_complete.npy",hac.hac(set2,hac.complete_linkage,2))
np.save("outs/hac_3_complete.npy",hac.hac(set3,hac.complete_linkage,2))
np.save("outs/hac_4_complete.npy",hac.hac(set4,hac.complete_linkage,4))


np.save("outs/hac_1_average.npy",hac.hac(set1,hac.average_linkage,2))
np.save("outs/hac_2_average.npy",hac.hac(set2,hac.average_linkage,2))
np.save("outs/hac_3_average.npy",hac.hac(set3,hac.average_linkage,2))
np.save("outs/hac_4_average.npy",hac.hac(set4,hac.average_linkage,4))


np.save("outs/hac_1_centroid.npy",hac.hac(set1,hac.centroid_linkage,2))
np.save("outs/hac_2_centroid.npy",hac.hac(set2,hac.centroid_linkage,2))
np.save("outs/hac_3_centroid.npy",hac.hac(set3,hac.centroid_linkage,2))
np.save("outs/hac_4_centroid.npy",hac.hac(set4,hac.centroid_linkage,4))