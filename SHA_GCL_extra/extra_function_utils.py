import numpy as np

def generate_current_predicate_set(incremental_stage_list, current_training_stage):
    outp = []
    formerp = []
    current_chosen_vector = []
    former_chosen_vector = []
    for i in range(current_training_stage + 1):
        outp.extend(incremental_stage_list[i])
    for i in range(current_training_stage):
        formerp.extend(incremental_stage_list[i])
    for i in range(len(outp)+1):
        if i in incremental_stage_list[current_training_stage]:
            current_chosen_vector.append(1)
        else:
            current_chosen_vector.append(0)
    for i in range(len(outp)+1):
        if i in formerp:
            former_chosen_vector.append(1)
        else:
            former_chosen_vector.append(0)
    num_stage_vector = []
    n_p = 0
    for isl in incremental_stage_list:
        n_p += len(isl)
        num_stage_vector.append(n_p)

    return outp, formerp, current_chosen_vector, former_chosen_vector, num_stage_vector

def generate_num_stage_vector(incremental_stage_list):
    num_stage_vector = []
    n_p = 0
    for isl in incremental_stage_list:
        n_p += len(isl)
        num_stage_vector.append(n_p)

    return num_stage_vector

def get_current_predicate_idx(incremental_stage_list, zeros_vector_penalty, Dataset_choice):
    data_long = 0
    if Dataset_choice == 'VG':
        data_long = 51
    elif Dataset_choice == 'GQA_200':
        data_long = 101
    else:
        exit('wrong mode in Dataset choice')
    outp = []
    for i in range(data_long):
        outp.append(0)
    for i in range(len(incremental_stage_list)):
        for num in incremental_stage_list[i]:
            outp[num] = i+1
    max_p = []
    for i in incremental_stage_list:
        max_p.append(max(i))

    idx_search_p = []
    kd_p = []
    for i in range(len(incremental_stage_list)):
        p1 = []
        p2 = []
        for j in range(data_long):
            p1.append(0)
            p2.append(zeros_vector_penalty)
        max_l = max_p[i]
        for j in range(max_l):
            p1[j+1] = j+1
            p2[j+1] = 1.0
        idx_search_p.append(p1)
        kd_p.append(p2)

    # for i in idx_search_p:
    #     print(i)
    # print()
    # for i in kd_p:
    #     print(i)
    return outp, max_p, idx_search_p, kd_p

def generate_onehot_vector(incremental_stage_list, current_training_stage, Dataset_choice):
    data_long = 0
    if Dataset_choice == 'VG':
        data_long = 51
    elif Dataset_choice == 'GQA_200':
        data_long = 101
    else:
        exit('wrong mode in Dataset choice')
    one_hot_vector = []
    if current_training_stage == -1:
        one_hot_vector.append(0)
        for i in range(data_long-1):
            one_hot_vector.append(1)
        return one_hot_vector
    for i in range(data_long):
        one_hot_vector.append(0)
    for i in range(current_training_stage+1):
        if i+1 == current_training_stage:
            for idx in incremental_stage_list[i]:
                if idx != 1 and idx != 2:
                    one_hot_vector[idx] = 1
                else:
                    one_hot_vector[idx] = -1
        elif i == current_training_stage:
            for idx in incremental_stage_list[i]:
                one_hot_vector[idx] = 1
        else:
            for idx in incremental_stage_list[i]:
                one_hot_vector[idx] = -1

    return one_hot_vector

def generate_sample_rate_vector(Dataset_choice, num_stage_predicate):
    if Dataset_choice == 'VG':
        predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712,
                                     5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352,
                                     663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270,
                                     234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
        assert len(predicate_new_order_count) == 51
    elif Dataset_choice == 'GQA_200':
        predicate_new_order_count = [200000, 64218, 47205, 32126, 25203, 21104, 15890, 15676, 7688, 6966, 6596, 6044, 5250, 4260, 4180, 4131, 2859, 2559, 2368, 2351, 2134, 1673, 1532, 1373, 1273, 1175, 1139, 1123, 1077, 941, 916, 849, 835, 808, 782, 767, 628, 603, 569, 540, 494, 416, 412, 412, 398, 395, 394, 390, 345, 327, 302, 301, 292, 275, 270, 267, 267, 264, 258, 251, 233, 233, 229, 224, 215, 214, 209, 204, 198, 195, 192, 191, 185, 181, 176, 158, 158, 154, 151, 148, 143, 136, 131, 130, 130, 128, 127, 125, 124, 124, 121, 118, 112, 112, 106, 105, 104, 103, 102, 52, 52]
        assert len(predicate_new_order_count) == 101
    else:
        exit('wrong mode in Dataset_choice')
    outp = []
    for i in range(len(num_stage_predicate)):
        opiece = []
        for j in range(len(predicate_new_order_count)):
            opiece.append(0.0)
        num_list = predicate_new_order_count[0:(num_stage_predicate[i]+1)]
        median = np.median(num_list[1:])
        for j in range(len(num_list)):
            if num_list[j] > median:
                num = median / num_list[j]
                if j == 0:
                    num = num * 10.0
                if num < 0.01:
                    num = 0.01
                opiece[j] = num
            else:
                opiece[j] = 1.0
        outp.append(opiece)
    return outp

def generate_current_group_sequence_for_bias(current_set, Dataset_choice):
    data_long = 0
    if Dataset_choice == 'VG':
        data_long = 51
    elif Dataset_choice == 'GQA_200':
        data_long = 101
    else:
        exit('wrong mode in Dataset choice')
    outp = []
    for i in range(data_long):
        outp.append(0)
    for i in current_set:
        outp[i] = i
    return outp

def generate_current_sequence_for_bias(incremental_stage_list, Dataset_choice):
    data_long = 0
    if Dataset_choice == 'VG':
        data_long = 51
    elif Dataset_choice == 'GQA_200':
        data_long = 101
    else:
        exit('wrong mode in Dataset choice')
    outp = []
    for i in range(len(incremental_stage_list)):
        opiece = []
        for j in range(data_long):
            opiece.append(0)
        for j in range(i+1):
            for k in incremental_stage_list[j]:
                opiece[k] = k
        outp.append(opiece)

    return outp