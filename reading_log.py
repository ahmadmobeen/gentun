import ast
generations=[]
current_gen=0
with open("tests/200326_csa_c20_i10_f3.txt" , 'r') as f:
    lines=f.readlines()
    for i,line in enumerate(lines):
        if "Running iteration" in line:
            current_gen=int(line.split('#')[1].split('.')[0])-1
            generations.append({"individuals":0,"accuracy":0,"fittest":{}})
        # if "Got fitness for" in line:
        #     generations[current_gen]["individuals"]+=1
        # if "S_1" in line:
        #     generations[current_gen]["fittest"]=ast.literal_eval(line)
        if "Fitness for" in line:
            generations[current_gen]["individuals"]+=1
        if "Crow with best location is:" in line:

            generations[current_gen]["fittest"] = ast.literal_eval(lines[i+1])
        if "Fitness value is" in line:
            generations[current_gen]["accuracy"]=float(line.split(':')[1])

print("Total Individuals:",len(generations)*generations[0]["individuals"])
print("Repeating Individuals:",len(generations)*generations[0]["individuals"]-sum([x["individuals"] for x in generations]))
print("New Individual Trainings:",sum([x["individuals"] for x in generations]))

print("Accuracy Achieved in First Generation:",generations[0]["accuracy"])
print("Maximum Accuracy Achieved:",max([x["accuracy"] for x in generations]))
print("Minimum Accuracy Achieved:",min([x["accuracy"] for x in generations]))
print("Average Accuracy Achieved: %.4f" % (sum([x["accuracy"] for x in generations])/len(generations)))
print("Accuracy Achieved in First Generation:",generations[len(generations)-1]["accuracy"])

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.interpolate as interp


ind = np.arange(len(generations))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, [x["individuals"] for x in generations], width)
plt.ylabel('Individuals')
plt.title('Individual Trainings Over Generataions')
# plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 21, 1))
plt.savefig('individuals.png')
plt.show()

plt.clf()
p1 = plt.plot(ind, [x["accuracy"] for x in generations])
plt.ylabel('Cross Validation Accuracy')
plt.title('Performance Over Generataions')
# plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0.997, 0.999, 0.001))

plt.savefig('individuals.png')
plt.show()

plt.clf()

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
# s1=int('111', 2)
# s2=int('1111111111', 2)
# X = np.arange(0, s1, 1)
# Y = np.arange(0, s2, 1)
#
#
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
#
# for i,x in enumerate(Z):
#     for j,y in enumerate(x):
#         Z[i,j]=0

X=[]
Y=[]
Z=[]
for g in generations:
    fittest=g["fittest"]
    acc=g["accuracy"]
    i=int(fittest["S_1"],2)
    j = int(fittest["S_2"], 2)
    # print(i,j)
    # Z[j-1][i-1]=acc
    X.append(i)
    Y.append(j)
    Z.append(acc)
print(Z)
plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),10),np.linspace(np.min(Y),np.max(Y),10))
plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')

# Plot the surface.
surf = ax.plot_surface(plotx, ploty, plotz, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0.9977, 0.9978)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()