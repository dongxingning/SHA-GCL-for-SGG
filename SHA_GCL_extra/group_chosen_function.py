# we rearrange the VG dataset, sort the relation classes in descending order (the original order is based on relation class names)
predicate_new_order = [0, 10, 42, 43, 34, 28, 17, 19, 7, 29, 33, 18, 35, 32, 27, 50, 22, 44, 45, 25, 2, 9, 5, 15, 26, 23, 37, 48, 41, 6, 4, 1, 38, 21, 46, 30, 36, 47, 14, 49, 11, 16, 39, 13, 31, 40, 20, 24, 3, 12, 8]
predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712, 5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352, 663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270, 234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
predicate_new_order_name = ['__background__', 'on', 'has', 'wearing', 'of', 'in', 'near', 'behind', 'with', 'holding', 'above', 'sitting on', 'wears', 'under', 'riding', 'in front of', 'standing on', 'at', 'carrying', 'attached to', 'walking on', 'over', 'for', 'looking at', 'watching', 'hanging from', 'laying on', 'eating', 'and', 'belonging to', 'parked on', 'using', 'covering', 'between', 'along', 'covered in', 'part of', 'lying on', 'on back of', 'to', 'walking in', 'mounted on', 'across', 'against', 'from', 'growing on', 'painted on', 'playing', 'made of', 'says', 'flying in']

def get_group_splits(Dataset_name, split_name):
    assert Dataset_name in ['VG', 'GQA_200']
    incremental_stage_list = None
    predicate_stage_count = None
    if Dataset_name == 'VG':
        assert split_name in ['divide3', 'divide4', 'divide5', 'average']
        if split_name == 'divide3':#[]
            incremental_stage_list = [[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9, 10, 11, 12, 13, 14],
                                      [15, 16, 17, 18, 19, 20],
                                      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
            predicate_stage_count = [3, 3, 8, 6, 20, 10]
        elif split_name == 'divide4':#[4,4,9,19,12]
            incremental_stage_list = [[1, 2, 3, 4],
                                      [5, 6, 7, 8, 9, 10],
                                      [11, 12, 13, 14, 15, 16, 17, 18, 19],
                                      [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
                                      [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
            predicate_stage_count = [4, 6, 9, 19, 12]
        elif split_name == 'divide5':
            incremental_stage_list = [[1, 2, 3, 4],
                                      [5, 6, 7, 8, 9, 10, 11, 12],
                                      [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                      [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
            predicate_stage_count = [4, 8, 10, 28]
        elif split_name == 'average':
            incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                                      [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
            predicate_stage_count = [10, 10, 10, 10, 10]
        else:
            exit('wrong mode in group split!')
        assert sum(predicate_stage_count) == 50

    elif Dataset_name == 'GQA_200':
        assert split_name in ['divide3', 'divide4', 'divide5', 'average']
        if split_name == 'divide3':  # []
            incremental_stage_list = [[1, 2, 3, 4],
                                      [5, 6, 7, 8],
                                      [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                      [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                                      [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [4, 4, 11, 16, 31, 34]
        elif split_name == 'divide4':  # [4,4,9,19,12]
            incremental_stage_list = [[1, 2, 3, 4, 5],
                                      [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                      [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                                      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [5, 10, 20, 65]
        elif split_name == 'divide5':
            incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7],
                                      [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                                      [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                                      [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [7, 14, 28, 51]
        elif split_name == 'average':
            incremental_stage_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                                      [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
                                      [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]]
            predicate_stage_count = [20, 20, 20, 20, 20]
        else:
            exit('wrong mode in group split!')
        assert sum(predicate_stage_count) == 100

    else:
        exit('wrong mode in group split!')

    return incremental_stage_list, predicate_stage_count