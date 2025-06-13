import torch

def cal_para(model):
    para_size = 0
    for name, para in model.named_parameters():
        size = 1
        id = 1
        for i in para.shape:
            if id == 1:
                this_size = i
            else:
                this_size = this_size * i
            id += 1
        para_size += this_size
    return (para_size / 1000000)
