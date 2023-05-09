import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.init import xavier_normal_, xavier_uniform_


class ConvE(nn.Module):
    def __init__(self, h_dim, out_channels, ker_sz):
        super().__init__()
        cfg = utils.get_global_config()
        self.cfg = cfg
        dataset = cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm1d(h_dim)

        self.conv_drop = torch.nn.Dropout(cfg.conv_drop)
        self.fc_drop = torch.nn.Dropout(cfg.fc_drop)
        self.k_h = cfg.k_h
        self.k_w = cfg.k_w
        assert self.k_h * self.k_w == h_dim
        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, stride=1, padding=0,
                                    kernel_size=ker_sz, bias=False)
        flat_sz_h = int(2 * self.k_h) - ker_sz + 1
        flat_sz_w = self.k_w - ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels
        self.fc = torch.nn.Linear(self.flat_sz, h_dim, bias=False)
        self.ent_drop = nn.Dropout(cfg.ent_drop_pred)

    def forward(self, head, rel, all_ent):
        # head (bs, h_dim), rel (bs, h_dim)
        # concatenate and reshape to 2D
        c_head = head.view(-1, 1, head.shape[-1])
        c_rel = rel.view(-1, 1, rel.shape[-1])
        c_emb = torch.cat([c_head, c_rel], 1)
        c_emb = torch.transpose(c_emb, 2, 1).reshape((-1, 1, 2 * self.k_h, self.k_w))

        x = self.bn0(c_emb)
        x = self.conv(x)  # (bs, out_channels, out_h, out_w)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_sz)  # (bs, out_channels * out_h * out_w)
        x = self.fc(x)  # (bs, h_dim)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc_drop(x)  # (bs, h_dim)
        # inference
        # all_ent: (n_ent, h_dim)
        all_ent = self.ent_drop(all_ent)
        x = torch.mm(x, all_ent.transpose(1, 0))  # (bs, n_ent)
        x = torch.sigmoid(x)
        return x


class DeepEBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_drop, activation=torch.nn.ReLU, layers=2, identity_drop=0):
        super(DeepEBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation()
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.final_bn = torch.nn.BatchNorm1d(output_dim)
        self.identity_bn = torch.nn.BatchNorm1d(output_dim)
        self.identity_drop = None
        if identity_drop != 0:
            self.identity_drop = torch.nn.Dropout(identity_drop)

        assert (layers >= 2)

        self.reslayer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.BatchNorm1d(output_dim)
        )

        self.reslayer.append(self.hidden_drop)
        #self.reslayer.append(activation())

        for i in range(layers - 1):
            self.reslayer.append(torch.nn.Linear(output_dim, output_dim))
            self.reslayer.append(torch.nn.BatchNorm1d(output_dim))
            self.reslayer.append(self.hidden_drop)
            if i != (layers - 2):
                self.reslayer.append(activation())

        if input_dim != output_dim:
            self.dim_map = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.input_dim != self.output_dim:
            identity = self.dim_map(x)
            identity = self.hidden_drop(identity)
            identity = self.identity_bn(identity)
        else:
            identity = x
        if (self.input_dim == self.output_dim) and self.identity_drop != 0:
            identity = self.identity_drop(identity)
        x = identity + self.reslayer(x)
        x = self.final_bn(x)
        return x


class ResNetBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_drop, activation=torch.nn.ReLU, layers=2):
        super(ResNetBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation()

        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.final_bn = torch.nn.BatchNorm1d(output_dim)
        self.identity_bn = torch.nn.BatchNorm1d(output_dim)
        self.hidden_drop = None
        if hidden_drop != 0:
            self.hidden_drop = torch.nn.Dropout(hidden_drop)

        assert (layers >= 2)

        self.reslayer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.BatchNorm1d(output_dim)
        )

        self.reslayer.append(activation())

        for i in range(layers - 1):
            self.reslayer.append(torch.nn.Linear(output_dim, output_dim))
            self.reslayer.append(torch.nn.BatchNorm1d(output_dim))

            if i != (layers - 2):
                self.reslayer.append(activation())

        if input_dim != output_dim:
            self.dim_map = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.input_dim != self.output_dim:
            identity = self.dim_map(x)
        else:
            identity = x
        x = identity + self.reslayer(x)
        x = self.final_bn(x)

        if self.hidden_drop is not None:
            x = self.hidden_drop(x)
        x = self.activation(x)

        return x


class DeepE(torch.nn.Module):
    def __init__(self, num_emb, embedding_dim=300, hidden_drop=0.4, num_source_layers=3, num_target_layers=1,
                 input_drop=0.4, inner_layers=3, target_drop=0.4, identity_drop=0.01):
        super(DeepE, self).__init__()
        cfg = utils.get_global_config()
        self.emb = torch.nn.Embedding(14541, embedding_dim)
        self.num_source_layers = num_source_layers
        self.num_target_layers = num_target_layers
        self.input_drop = torch.nn.Dropout(input_drop)
        self.input_bn = torch.nn.BatchNorm1d(2 * embedding_dim)
        self.target_bn = torch.nn.BatchNorm1d(embedding_dim)
        self.loss = torch.nn.CrossEntropyLoss()
        self.source_layers = torch.nn.Sequential()
        self.target_layers = torch.nn.Sequential()
        for i in range(num_source_layers):
            if i == 0:
                input_emb = embedding_dim * 2
            else:
                input_emb = embedding_dim
            self.source_layers.append(
                DeepEBlock(input_emb, embedding_dim, hidden_drop, torch.nn.ReLU, layers=2, identity_drop=identity_drop))
        '''
        for i in range(num_target_layers):
            self.target_layers.append(
                ResNetBlock(embedding_dim, embedding_dim, target_drop, torch.nn.ReLU, layers=inner_layers))
        '''
        #self.register_parameter('b', Parameter(torch.zeros(num_emb)))
        self.ent_drop = nn.Dropout(cfg.ent_drop_pred)

    def to_var(self, x, use_gpu=True):
        if use_gpu:
            return Variable(torch.from_numpy(x).long().cuda())

    def init(self):
        xavier_normal_(self.emb.weight.data)
        self.emb.weight.data = self.emb.weight.data

    def forward(self, e1, rel,all_ent):

        e1_embedded = e1
        rel_embedded = rel

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], -1)

        x = self.input_bn(stacked_inputs)
        x = self.input_drop(x)

        x = self.source_layers(x)
        '''
        weight = self.emb.weight

        weight = self.target_bn(weight)
        weight = self.target_layers(weight)
        # weight = self.input_drop(weight)
        weight = weight.transpose(1, 0)
        x = torch.mm(x, weight)

        y = self.b
        y1 = y.expand_as(x)
        x += y1

        '''
        all_ent = self.ent_drop(all_ent)
        # x:Tensor(2,450),all_ent.transpose:Tensor(450,14541)------>Tensor(2,14541)
        x = torch.mm(x, all_ent.transpose(1, 0))  # (bs, n_ent)

        x=torch.sigmoid(x)

        return x

