import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv


# class GCN(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, num_classes)
#
#     def forward(self, g, in_feat):
#         h = self.conv1(g, in_feat)
#         h = F.relu(h)
#         h = self.conv2(g, h)
#         return h

class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = GraphConv(in_feats=in_feats, out_feats=hid_feats, bias=False, allow_zero_in_degree=True)
        self.conv1 = GraphConv(in_feats=hid_feats, out_feats=hid_feats, bias=False, allow_zero_in_degree=True)
        self.conv2 = GraphConv(in_feats=hid_feats, out_feats=out_feats, bias=False, allow_zero_in_degree=True)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean', bias=False)
        self.conv1 = SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean', bias=False)
        self.conv2 = SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean', bias=False)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h