import matplotlib.pyplot as plt
import numpy as np
diff=[1,2,3,4,5,6,7,8,9,10,11,12,13]
fl=[int(np.round(np.sqrt(13*x-9))) for x in diff]
# fl=[np.sqrt(13*x-9) for x in diff]

print(diff)
print(fl)
p1 = plt.plot(diff, fl)
plt.ylabel('Flight Length')
plt.xlabel('Distance from Target')
plt.title('Flight Length Normalization')
# plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.xticks(diff)
plt.yticks(np.arange(0, len(diff)+2, 1))

# plt.savefig('200407_csa_20i_20c_fl13_ap15.png')
plt.show()