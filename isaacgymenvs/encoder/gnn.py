from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch
import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool
from torch_cluster import fps

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,device):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels*2 + 3, out_channels,device=device),
                              ReLU(),
                              Linear(out_channels, out_channels,device=device))

    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_i, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_i, h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.

class PointNet(torch.nn.Module):
    def __init__(self,device):
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(1, 32,device=device)
        self.conv2 = PointNetLayer(32, 32,device=device)
        self.conv3 = PointNetLayer(32, 32,device=device)

        self.regression = Linear(32, 10,device=device)

    def forward(self,x, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.

        edge_index = knn_graph(pos, k=6, batch=batch, loop=True)
        h = self.conv1(h=x, pos=pos, edge_index=edge_index)
        h = h.relu()
        index = fps(pos, batch=batch, ratio=0.5)
        pos = pos[index]
        h = h[index]
        batch = batch[index]

        edge_index = knn_graph(pos, k=4, batch=batch, loop=True)
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        index = fps(pos, batch=batch, ratio=0.5)
        pos = pos[index]
        h = h[index]
        batch = batch[index]

        edge_index = knn_graph(pos, k=3, batch=batch, loop=True)
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        o = self.regression(h)

        return o

