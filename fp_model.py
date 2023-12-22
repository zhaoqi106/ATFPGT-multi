import torch
from torch import nn


class MolFPEncoder(nn.Module):
    def __init__(self, emb_dim, drop_ratio, fp_type, device):
        super(MolFPEncoder, self).__init__()
        self.fp_type = fp_type
        self.device = device
        morgan_dim = 2048 if 'morgan' in fp_type else 0
        maccs_dim = 167 if 'maccs' in fp_type else 0
        rdit_dim = 2048 if 'rdit' in fp_type else 0

        init_dim = morgan_dim + maccs_dim + rdit_dim

        self.fc1 = nn.Linear(init_dim, emb_dim).to(device)
        self.batch_norm = nn.BatchNorm1d(emb_dim).to(device)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_ratio)
        self.fc2 = nn.Sequential(
            nn.Linear(init_dim, 512).to(device),
            nn.Dropout(),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.init_emb()
        self.sigmoid = nn.Sigmoid()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        morgan_fp = data.morgan_fp.to(self.device) if 'morgan' in self.fp_type else torch.empty(0)
        maccs_fp = data.maccs_fp.to(self.device) if 'maccs' in self.fp_type else torch.empty(0)
        rdit_fp = data.rdit_fp.to(self.device) if 'rdit' in self.fp_type else torch.empty(0)

        fps = torch.cat([morgan_fp, maccs_fp, rdit_fp], dim=1)
        fps_rep = self.act_func(self.batch_norm(self.fc2(fps.float())))

        return fps_rep
