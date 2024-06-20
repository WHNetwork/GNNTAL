import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv,SAGEConv

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class GNNT(nn.Module):
    def __init__(self, SAGE_P, Transformer_P):
        super(GNNT, self).__init__()
        # GraphSAGE layers
        self.sage1 = SAGEConv(SAGE_P["in_dim"], SAGE_P["embed_dim"],aggregator_type="lstm")
        self.sage2 = SAGEConv(SAGE_P["embed_dim"], SAGE_P["embed_dim"],aggregator_type="lstm")

        # The hidden dimension of GCN layers should match the d_model of the Transformer
        d_model = SAGE_P["embed_dim"]

        # Transformer layers
        self.transformer_layer = nn.Transformer(
            d_model=d_model,
            nhead=Transformer_P["nhead"],
            num_encoder_layers=Transformer_P["num_encoder_layers"],
            num_decoder_layers=Transformer_P["num_decoder_layers"],
            dim_feedforward=Transformer_P["dim_feedforward"],
            dropout=Transformer_P["dropout"]
        )

        # Linear layers
        self.fc1 = nn.Linear(d_model, 16)
        self.fc2 = nn.Linear(16, 1)
        self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)

        # Activation function
        activation_functions = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU()
        }
        self.activation = activation_functions[SAGE_P["activation"]]

    def forward(self, g, input_features):
        # GCN layers
        x = self.sage1(g, input_features)
        x = F.relu(x)
        x = self.sage2(g, x)
        x = F.relu(x)

        # Transformer layers
        x = x.unsqueeze(1)  # Adding a sequence dimension
        x = self.transformer_layer(x, x)
        x = x.squeeze(1)  # Removing the sequence dimension

        # Linear layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def reset_parameters(self):
        self.sage1.reset_parameters()
        self.sage2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.weight = nn.init.normal_(self.fc2.weight, 0.1, 0.01)


