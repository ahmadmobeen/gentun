import random



def old_method(bin_xi,bin_mj):
    diff = int(bin_mj, 2) - int(bin_xi, 2)
    if diff < 0:
        diff = diff * -1
    bin_diff = str(bin(diff >> 1) + str(diff & 1)).replace("b", "")

    if len(bin_diff) < len(bin_xi):
        for i in range(len(bin_xi) - len(bin_diff)):
            bin_diff = "0" + bin_diff

    fl = random.randrange(0, len(bin_xi), 1)

    index = random.sample(range(0, 14), fl)

    bin_distance = ""
    for i, c in enumerate(bin_xi):
        if i in index:
            bin_distance += bin_diff[i]
        else:
            bin_distance += "0"

    int_xiplus1 = int(bin_xi, 2) + int(bin_distance, 2)
    bin_xiplus1 = str(bin(int_xiplus1 >> 1) + str(int_xiplus1 & 1)).replace("b", "")
    if len(bin_xiplus1) > len(bin_xi):
        len_diff = len(bin_xiplus1) - len(bin_xi)
        bin_xiplus1 = bin_xiplus1[len_diff:]

    return bin_xiplus1

def new_method(bin_xi,bin_mj):
    # new_diff = [bin_mj[i] for i in range(len(bin_xi)) if bin_xi[i] != bin_mj[i]]

    diff_pos=[]
    for bit_pos,bit_xi in enumerate(bin_xi):
        if bit_xi!=bin_mj[bit_pos]:
            diff_pos.append(bit_pos)
    print(diff_pos)
    fl = random.randrange(0, len(bin_xi), 1)

    print("Flight_Length",fl)
    if fl>len(diff_pos):
        index = random.sample([x for x in range(0, len(bin_xi)) if x not in diff_pos], fl-len(diff_pos))
        diff_pos.extend(index)
    else:
        diff_pos=random.sample(diff_pos, fl)
    print(diff_pos)

    bin_xiplus1=list(bin_xi)
    for i in diff_pos:
        if bin_xiplus1[i]=='1':
            bin_xiplus1[i]='0'
        else:
            bin_xiplus1[i]='1'


    bin_xiplus1="".join(bin_xiplus1)

    print(bin_xi)
    print(bin_mj)
    print(bin_xiplus1)



    return bin_xiplus1




if __name__=="__main__":
    new=True

    new_location={}
    current_location = {"S_1":"010","S_2":"1001100111"}
    target_location = {"S_1":"110","S_2":"1011000101"}
    bin_xi = "".join([current_location[stage] for stage in current_location.keys()])
    bin_mj = "".join([target_location[stage] for stage in target_location.keys()])



    if not new:
        bin_xiplus1=old_method(bin_xi,bin_mj)
    else:
        bin_xiplus1 = new_method(bin_xi, bin_mj)

    # last = 0
    # for name, connections in {"S_1":3,"S_2":10}.items():
    #     end = last + connections
    #     bit_string = bin_xiplus1[last:end]
    #     if len(bit_string) < connections:
    #         for bit in range(connections - len(bit_string)):
    #             bit_string = "0" + bit_string
    #     new_location[name] = bit_string
    #     last = end
    #
    #
    # print(current_location)
    # print(target_location)
    # print(new_location)