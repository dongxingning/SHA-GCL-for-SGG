import json
from SHA_GCL_extra.dataset_path import datasets_path
from SHA_GCL_extra.get_your_own_group.basic_for_all_test import indeed_train_sor, predicate_dict

def generate_groups_by_n_times(times=4):
    '''
    Get your own groups!
    For every element in group, the maximal amount of training instances will be no more than x times of the minimal
    '''
    group_list = []
    shunxu_list = []
    clist = []
    slist = []
    counting_list = []
    head_num = indeed_train_sor[0][1]
    end_num = int(head_num/times)
    idx = 0
    for name, data in indeed_train_sor:
        idx += 1
        if data >= end_num or end_num < 200:
            clist.append(predicate_dict[name])
            slist.append(idx)
        else:
            counting_list.append([head_num, end_num])
            head_num = data
            end_num = int(data/times)
            group_list.append(clist)
            shunxu_list.append(slist)
            clist = [predicate_dict[name]]
            slist = [idx]
    counting_list.append([head_num, end_num])
    group_list.append(clist)
    shunxu_list.append(slist)
    num_count_list = []
    for i in shunxu_list:
        num_count_list.append(len(i))
    print(num_count_list)
    print(shunxu_list)
    print(group_list)
    print(counting_list)
    return group_list

def generate_groups_by_n_times_GQA(ID_info, times=3):
    with open(ID_info, 'r') as f:
        data_json = json.load(f)
    rel_count = data_json['rel_count']
    id_to_predicate = data_json['rel_name_to_id']
    id_to_predicate.pop('__background__')
    rel_count.pop('__background__')
    rel_count = sorted(rel_count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    group_list = []
    shunxu_list = []
    clist = []
    slist = []
    counting_list = []
    head_num = rel_count[0][1]
    end_num = int(head_num/times)
    idx = 0
    for rel_data in rel_count:
        idx += 1
        name, count = rel_data
        if count >= end_num or end_num < 200:
            clist.append(id_to_predicate[name])
            slist.append(idx)
        else:
            counting_list.append([head_num, end_num])
            head_num = count
            end_num = int(count/times)
            group_list.append(clist)
            shunxu_list.append(slist)
            clist = [id_to_predicate[name]]
            slist = [idx]
    counting_list.append([head_num, end_num])
    group_list.append(clist)
    shunxu_list.append(slist)
    num_count_list = []
    for i in shunxu_list:
        num_count_list.append(len(i))
    print(num_count_list)
    print(shunxu_list)
    print(group_list)
    print(counting_list)
    return group_list

if __name__ == '__main__':
    ID_info = datasets_path + 'datasets/gqa/GQA_200_ID_Info.json'
    generate_groups_by_n_times(times=4)
    print('\n\n')
    generate_groups_by_n_times_GQA(ID_info, times=4)