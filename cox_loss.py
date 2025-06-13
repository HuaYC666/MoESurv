import torch
import torch.nn as nn
'''
define cox loss 
'''

def coxloss_mask(OStime, event_id):
    n_sample = len(OStime)
    mask_mat = torch.ones([n_sample, n_sample])
    # 构建发生event结局的患者的生存时间
    event_OStime = []
    for i in event_id:
        event_OStime.append(OStime[i])
    # 对掩码矩阵进行零点处理
    for id in event_id:
        OStime_for_id = OStime[id]  # 第id个患者的生存时间
        if OStime[id:].count(OStime_for_id) > 1:  # 即若存在同生存时间的其他患者
            sametime_id = [index for index, value in enumerate(OStime) if
                           value == OStime_for_id and index > id and index in event_id]
            mask_mat[id, sametime_id] = 0
    return mask_mat


class coxloss(nn.Module):
    def __init__(self, OStime, event_id, device):
        super(coxloss, self).__init__()
        self.event_id = event_id
        self.mask = coxloss_mask(OStime, event_id).to(device)

    def forward(self, x):
        x_adjust = x
        x_exp = torch.exp(x_adjust)
        n_sample = x.shape[0]
        x_exp = x_exp.unsqueeze(0)
        x_exp_mat = x_exp.repeat(n_sample, 1)

        # 从risk中抽取对角线元素
        x_risk = x_adjust[self.event_id]

        # 构建上三角矩阵
        x_tril_mat = torch.triu(x_exp_mat, 1)  # 不要对角线元素
        x_tril_mat = torch.mul(x_tril_mat, self.mask)  # 对同时间事件进行掩码
        x_tril_mat_rowsum = torch.sum(x_tril_mat, 1)
        x_tril_mat_logrowsum = torch.log(x_tril_mat_rowsum[self.event_id][:-1])

        loss = - x_risk.sum() + x_tril_mat_logrowsum.sum()

        return loss  # ,x_exp,x_tril_mat_rowsum,x_tril_mat_logrowsum,x_risk,x_tril_mat

# define vae loss
def vae_loss_function(recon_x, x, mu, logvar):
    """VAE loss = reconstructed loss（MSE） + KL divergence"""
    MSE = torch.nn.functional.mse_loss(recon_x, x.view(-1, x.size(1)), reduction='sum')  
    # KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE, KLD, MSE + KLD
