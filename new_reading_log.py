import json
import matplotlib.pyplot as plt
import numpy as np


data={
    "flock_size":0,
    "total_iterations":0,
    "initial_flock":[],
    "iterations":[]
}
init=False
id=0
with open("200407_csa_20i_20c_fl13_ap15.txt" , 'r') as f:
    lines=f.readlines()
    for i,line in enumerate(lines):
        if "Initializing a random flock." in line:
            data["flock_size"]=int(line.split(":")[1])
        if "S_1" in line and not init:
            id =line.split(" ")[0]
            data["initial_flock"].append({id:json.loads(line.replace(id+" ","").replace("\'", "\""))})
        if "Starting Crow Search Algorithm..." in line:
            init=True
        if "Running iteration #" in line:
            data["total_iterations"]+=1
            data["iterations"].append({"best_location":{},"best_acc":0,"best_crow":"0","crows":[]})
        if "Evaluating individual" in line:
            id = line.split(" ")[4]
            location=json.loads("{"+line.split("{")[1].split("}")[0].replace("\'", "\"")+"}")
            data["iterations"][data["total_iterations"]-1]["crows"].append({"id":id,"location":location})
        if "Fitness of Crow " in line:
            # id = line.split(" ")[5]
            last_location=json.loads("{"+line.split("{")[1].split("}")[0].replace("\'", "\"")+"}")
            last_acc=float(line.split("was")[1].split(" ")[1])
            for crow in data["iterations"][data["total_iterations"]-1]["crows"]:
                if crow["id"]==id:
                    pass
                    # crow["last_location"]=last_location
                    # crow["last_acc"]=last_acc
        if "Best known performance of" in line:
            # id = line.split(" ")[7]
            best_acc=float(line.split("is ")[1].split(" on")[0])
            memory=json.loads(line.split("location ")[1].replace("\'", "\""))
            for crow in data["iterations"][data["total_iterations"]-1]["crows"]:
                if crow["id"]==id:
                    pass
                    # if data["total_iterations"]>1:
                    #     for crow_in_last_iter in data["iterations"][data["total_iterations"]-2]["crows"]:
                    #         if crow_in_last_iter["id"]==id:
                    #             if "memory_acc" in crow_in_last_iter.keys():
                                    # assert (crow_in_last_iter["memory_acc"] == best_acc)
                                    # assert (crow_in_last_iter["memory_location"] == memory)
                    # crow["last_memory_acc"]=best_acc
                    # crow["last_memory_location"]=memory
        if "Performance of Crow" in line:
            # id = line.split(" ")[5]
            acc = float(line.split("is ")[1].split(" on")[0])
            location=json.loads(line.split("location ")[1].replace("\'", "\""))
            for crow in data["iterations"][data["total_iterations"]-1]["crows"]:
                if crow["id"]==id:
                    assert (crow["location"]==location)
                    crow["acc"]=acc

        if "remains the same" in line:
            best_acc=float(line.split(" ")[13])
            memory = json.loads(line.split("location ")[1].replace("\'", "\""))
            for crow in data["iterations"][data["total_iterations"]-1]["crows"]:
                if crow["id"]==id:
                    crow["memory_acc"]=best_acc
                    crow["memory_location"]=memory

        if "Updating best known performance" in line:
            # id = line.split(" ")[8]
            best_acc = float(line.split("to ")[1].split(" on")[0])
            memory = json.loads(line.split("location ")[1].replace("\'", "\""))
            for crow in data["iterations"][data["total_iterations"]-1]["crows"]:
                if crow["id"]==id:
                    crow["memory_acc"]=best_acc
                    crow["memory_location"]=memory

        if "is following the Crow" in line:
            id = line.split(" ")[4]
            target=line.split("is following the Crow")[1].split(" on location")[0]
            for crow in data["iterations"][data["total_iterations"]-1]["crows"]:
                if crow["id"]==id:
                    crow["target"]=target

        if "being followed by the Crow " in line:
            # id = line.split(" ")[13]
            if "not" in line:
                aware=False
            else:
                aware=True

            for crow in data["iterations"][data["total_iterations"]-1]["crows"]:
                if crow["id"]==id:
                    crow["aware"]=aware

        if "The flight length of Crow " in line:
            flight_length= int(line.split(" ")[11])
            for crow in data["iterations"][data["total_iterations"]-1]["crows"]:
                if crow["id"]==id:
                    crow["flight_length"]=flight_length

        if "reaches a new location" in line:
            new_location=json.loads(line.split("location  ")[1].split(" .")[0].replace("\'", "\""))
            for crow in data["iterations"][data["total_iterations"]-1]["crows"]:
                if crow["id"]==id:
                    pass
                    # crow["new_location"]=new_location

        if "Best performance is" in line:
            data["iterations"][data["total_iterations"] - 1]["best_acc"]=float(line.split(" ")[3])
            data["iterations"][data["total_iterations"] - 1]["best_crow"]=line.split(" ")[6]
            data["iterations"][data["total_iterations"] - 1]["best_location"]=json.loads(line.split("the location : ")[1].replace("\'", "\""))
            print(data["total_iterations"])

# print("Final",data["total_iterations"])
#
# final_best_crow = sorted(data["iterations"][data["total_iterations"]-1]["crows"], key=lambda k: k['memory_acc'],reverse=True)[0]
#
# data["iterations"][data["total_iterations"] - 1]["best_acc"] = final_best_crow["memory_acc"]
# data["iterations"][data["total_iterations"] - 1]["best_crow"] = final_best_crow["id"]
# data["iterations"][data["total_iterations"] - 1]["best_location"] = final_best_crow["memory_location"]
#
#
# print(len(data["iterations"]))


for id in range(data["flock_size"]):
    id=str(id)
    print("\nCrow", id)
    for iteration in range(data["total_iterations"]):
        for crow in data["iterations"][iteration]["crows"]:
            if crow["id"]==id:
                print("Iteration",iteration,":",crow)



print("\n\nResults")
for iteration in range(data["total_iterations"]):
    print("Iteration",iteration,":","Best Crow:",data["iterations"][iteration]["best_crow"],"Accuracy:",data["iterations"][iteration]["best_acc"],"Location:",data["iterations"][iteration]["best_location"])





ind = np.arange(data["total_iterations"])    # the x locations for the groups
# width = 0.35       # the width of the bars: can also be len(x) sequence
# p1 = plt.bar(ind, [x["individuals"] for x in data["total_iterations"]], width)
# plt.ylabel('Crow Individuals')
# plt.title('Crow Individual Trainings Over Iterataions')
# # plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
# plt.yticks(np.arange(0, 21, 1))
# plt.savefig('individuals.png')
# plt.show()
#
# plt.clf()
p1 = plt.plot(ind, [x["best_acc"] for x in data["iterations"] if x["best_acc"]>0])
plt.ylabel('Cross Validation Accuracy')
plt.xlabel('Number of Iterations')
plt.title('Performance Over Iterataions')
# plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.xticks(np.arange(0, data["total_iterations"], 1))
plt.yticks(np.arange(0.910, 0.917, 0.001))

plt.savefig('200407_csa_20i_20c_fl13_ap15.png')
plt.show()








# print(json.dumps(data,indent=2))