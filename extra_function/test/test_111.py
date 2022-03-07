import torch
import numpy as np

if __name__ == '__main__':
    aaa_path = '/home/share/dongxingning/important_savings/2022CVPR/2022CVPR_models/MHA/GQA/PredCls_MHAGCL/inference_best/GQA_Simple_test/result_dict.pytorch'
    a2_path = '/home/share/dongxingning/important_savings/2022CVPR/2022CVPR_models/MHA/VG/PredCls_MHAGCL/inference/60000_best/result_dict.pytorch'
    aaa = torch.load(aaa_path)
    print(aaa.keys())
    print(aaa['predcls_mean_recall_list'][100])
    print(len(aaa['predcls_mean_recall_list'][100]))
    print(np.mean(aaa['predcls_mean_recall_list'][100][100:]))
    print(np.mean(aaa['predcls_mean_recall_list'][100]))
    print(np.mean(aaa['predcls_mean_recall_list'][100][:100]))