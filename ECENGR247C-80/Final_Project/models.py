import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self,data,hidden=25):
        super(GCN,self).__init__()
        self.conv1 = GCNConv(data.num_node_features,hidden,cached=True)
        self.conv2 = GCNConv(hidden,data.num_classes,cached=True)
        
    def forward(self,data):
        x,edge_index,edge_weight = data.x,data.edge_index,data.edge_weight
        
        x = self.conv1(x,edge_index,edge_weight)
        x = F.relu(x)
        x = self.conv2(x,edge_index,edge_weight)
        
        return F.log_softmax(x,dim=1)