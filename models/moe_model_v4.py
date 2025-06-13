import torch
import numpy as np
import pandas as pd
import torch.nn as nn

'''
MLP conctruct
x -> input
output -> [n_sample，n_embeding_output]
'''
class MLP(nn.Module):
    def __init__(self, n_embeding_input,n_embeding_output):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(n_embeding_input, 2*n_embeding_output)
        self.w2 = nn.Linear(2*n_embeding_output, n_embeding_output)
        #self.w3 = nn.Linear(n_embeding_input, n_embeding_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.w1(x))
        x = self.relu(self.w2(x))
        return x

'''
gate layer
'''
class Gate(nn.Module):
    def __init__(self, n_embeding_input,n_embeding_output,top_k,n_expert_group,top_k_group,device):
        super(Gate, self).__init__()
        self.top_k = top_k
        self.n_expert_group = n_expert_group
        self.top_k_group = top_k_group
        self.linear = nn.Linear(n_embeding_input, n_embeding_output)
        self.sigmoid = nn.Sigmoid()
        self.gamma = 0.01
        self.bias = torch.zeros([1,n_embeding_output]).to(device)
        self.n_embeding_output = n_embeding_output

    def forward(self, x):
        # Note that score is used to calculate which routing expert to train, and score2weights determine the weights assigned to forward propagation
        score = self.linear(x)
        score2weights = self.sigmoid(score)
        score = self.sigmoid(score) + self.bias.to(x.device)

        score = score.view(x.size(0), self.n_expert_group, -1)
        score2weights = score2weights.view(x.size(0), self.n_expert_group, -1)

        group_score = score.topk(min(2, self.top_k), dim=-1)[0].sum(dim=-1)  # 得到每一组中前top_k得分之和
        indices = group_score.topk(self.top_k_group, dim=-1)[1]  # 获取前topk组的编号
        # block all values of unselected groups
        mask = torch.zeros_like(score[..., 0], device=x.device).scatter_(1, indices, True)
        score = (score * mask.unsqueeze(-1)).flatten(1)
        score2weights = (score2weights * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(score, self.top_k, dim=-1)[1]  # 从留下来的值中选topk个专家的编号
        weights = score2weights.gather(1, indices)

        # dynamics adjust
        counts = torch.bincount(indices.flatten(), minlength=self.n_embeding_output).tolist()
        id_down = [i for i in range(len(counts)) if counts[i] > np.percentile(counts, 75)]
        id_up = [i for i in range(len(counts)) if counts[i] < np.percentile(counts, 25)]
        self.bias[:, id_down] -= self.gamma
        self.bias[:, id_up] += self.gamma

        return weights,indices


'''
Encoder of MVAE
'''
class MOE(nn.Module):
    def __init__(self,n_disease,n_share,n_private,n_transfer,top_k,n_expert_group,top_k_group,dim_input,dim_output,mlp,gate,device):
        super(MOE, self).__init__()
        self.n_share = n_share
        self.n_private = n_private
        self.n_transfer = n_transfer
        self.n_expert_group = n_expert_group
        self.top_k = top_k
        self.moe_share = nn.ModuleList([mlp(dim_input,dim_output).to(device) for i in range(n_share)])
        self.moe_private = [nn.ModuleList([mlp(dim_input,dim_output).to(device) for i in range(n_private)]) for j in range(n_disease)]
        self.moe_transfer = nn.ModuleList([mlp(dim_input,dim_output).to(device) for i in range(n_transfer)])
        self.device = device
        self.dim_output = dim_output
        self.gate = gate(dim_input,n_transfer,top_k,n_expert_group,top_k_group,device)
        self.dim_output = dim_output

        self.linear_u_share = nn.Linear(dim_output,dim_output)
        self.linear_logvar_share = nn.Linear(dim_output, dim_output)

        self.linear_u_private = nn.Linear(dim_output, dim_output)
        self.linear_logvar_private = nn.Linear(dim_output, dim_output)

        self.linear_u_transfer = nn.Linear(dim_output, dim_output)
        self.linear_logvar_transfer = nn.Linear(dim_output, dim_output)

    def forward(self,x,disease_id):
        n_sample,n_dim = x.shape
        feature_share = torch.zeros([n_sample,self.dim_output],device=self.device)

        if self.n_share != 0:
            for i in range(self.n_share):
                feature_share = feature_share + self.moe_share[i](x)

        feature_private = torch.zeros([n_sample,self.dim_output],device=self.device)
        if self.n_private != 0:
            for i in range(n_sample):
                for j in range(self.n_private):
                    feature_private[i,:] = feature_private[i,:] + self.moe_private[disease_id[i]][j](x[i,:].unsqueeze(0)).squeeze(0)

        feature_transfer = torch.zeros([n_sample,self.dim_output],device=self.device)
        if self.n_transfer != 0:
            weights,indices = self.gate(x)
            #统计在所有batch里，每个专家被选择了多少次
            counts = torch.bincount(indices.flatten(),minlength=self.n_transfer).tolist()
            for i in range(self.n_transfer):
                if counts[i] == 0:
                    continue
                expert = self.moe_transfer[i]
                idx,top = torch.where(indices == i)
                feature_transfer[idx] += expert(x[idx])*weights[idx,top,None]

        u_share = self.linear_u_share(feature_share)
        logvar_share = self.linear_logvar_share(feature_share)

        u_private = self.linear_u_private(feature_private)
        logvar_private = self.linear_logvar_private(feature_private)

        u_transfer = self.linear_u_transfer(feature_transfer)
        logvar_transfer = self.linear_logvar_transfer(feature_transfer)


        return u_share+u_transfer+u_private,logvar_share+logvar_transfer+logvar_private



'''
MVAE
'''
class MOEsurv_net(nn.Module):
    def __init__(self,n_disease,n_share,n_private,n_transfer,top_k,n_expert_group,top_k_group,dim_input,dim_latent,n_layer,moe,mlp,gate,device):
        super(MOEsurv_net, self).__init__()
        self.moe = moe(n_disease,n_share,n_private,n_transfer,top_k,n_expert_group,top_k_group,dim_input,dim_latent,mlp,gate,device).to(device)
        self.linear = nn.Linear(dim_input,dim_latent*2)
        self.linear_output = nn.Linear(dim_latent,1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.n_layer = n_layer
        self.dropout = nn.Dropout(0.2)
        self.linear_recon = nn.Linear(dim_latent,dim_input)

        self.n_share = n_share
        self.n_private = n_private
        self.n_transfer = n_transfer

        self.decoder_linear_1 = nn.Linear(dim_latent,2*dim_latent)
        self.decoder_linear_2 = nn.Linear(2*dim_latent,dim_input)

    def forward(self,x,disease_id):
        mu,logvar = self.moe(x, disease_id)

        #开始重参数化设置
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        embeding = mu + eps * std

        recon = self.relu(self.decoder_linear_1(embeding))
        recon = self.decoder_linear_2(recon)


        return mu,embeding,recon
