import os
import random
from torch.utils.data import DataLoader, TensorDataset
from models.moe_model_v4 import MOEsurv_net,MOE,MLP,Gate
import torch
import pandas as pd
import numpy as np
from batch_construct import batch_construct,batch_construct_certrain_disease
from cal_para import cal_para
import torch.optim as optim
from cox_loss import vae_loss_function
from evaluation import CIndex


# define train function
def train(epoch, model, dataloader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        mu, logvar,recon_batch = model(data[0].float(),data[1])
        MSE,KLD,loss = vae_loss_function(recon_batch, data[0], mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')




expr = pd.read_csv("/data1/wyh/bulk_moe/Dataset/TCGA_pan_cancer/exp_data_raw.csv.gz")
expr = expr.drop(columns=["Unnamed: 0"])
sample_id = expr.columns.tolist()
expr = expr.T
expr = np.asarray(expr)
expr = torch.tensor(expr,dtype=torch.float32)
std = torch.std(expr,0).tolist()
no_zero_id = [i for i in range(len(std)) if std[i] !=0]
expr = expr[:,no_zero_id]
expr = torch.log2(expr+1)

# initialize MVAE model
surv = pd.read_csv("/data1/wyh/bulk_moe/Dataset/TCGA_pan_cancer/surv_data.csv.gz")
sample_id = surv["sample"].tolist()
disease_id = surv["cancer"].tolist()

all_disease = ["BRCA","KIRC","LUAD","HNSC","LUSC","LGG","SKCM","STAD","OV","BLCA","LIHC","COAD","KIRP","CESC","SARC","ESCA","UCEC","PAAD","LAML","GBM"]
disease_id = [all_disease.index(i) if i in all_disease else 999 for i in disease_id ]
disease_id = torch.tensor(disease_id)


OS = surv["OS"].tolist()
OStime = surv["OS.time"].tolist()

#dataset select
expr = expr[disease_id != 999,:]
select_sample_id = [sample_id[i] for i in range(disease_id.shape[0]) if disease_id[i] != 999]
disease_id = disease_id[disease_id != 999]

device = torch.device("cuda:0")
dataset = TensorDataset(torch.tensor(expr, dtype=torch.float32).to(device),disease_id.to(device))
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
n_disease = len(all_disease)
n_share = 1
n_private =1
n_transfer = 12
top_k = 2
n_expert_group=4
top_k_group=2
dim_input = expr.shape[1]
dim_latent = 256
n_layer=2

model = MOEsurv_net(n_disease=n_disease, n_share=n_share, n_private=n_private, n_transfer=n_transfer,
                    top_k=top_k,n_expert_group=n_expert_group,top_k_group=top_k_group,
                    dim_input=dim_input,dim_latent=dim_latent,n_layer=n_layer,
                    moe=MOE,mlp=MLP,gate=Gate,device=device)
model.to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# train model
num_epochs = 150
for epoch in range(1, num_epochs + 1):
    train(epoch, model, dataloader, optimizer)

torch.save(model,"/data1/wyh/bulk_moe/Dataset/TCGA_pan_cancer/model_pkl/VAE_moe_v4_256.pkl")

# Extracting latent features using trained model
model.eval()
with torch.no_grad():
    for i in range(len(model.moe_modules)):
        model.moe_modules[i].gate.bias = model.moe_modules[i].gate.bias - model.moe_modules[i].gate.bias
    data_tensor = torch.tensor(expr, dtype=torch.float32)
    mu,_, _ = model(data_tensor.to(device),disease_id.to(device))

latent_features = np.asarray(mu.detach().cpu())
latent_features = pd.DataFrame(latent_features)
latent_features.index = select_sample_id
latent_features.to_csv("/data1/wyh/bulk_moe/Dataset/TCGA_pan_cancer/vae_moe_v4_embeding_256.csv")

