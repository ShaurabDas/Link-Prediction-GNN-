import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, Sequential
import torch.nn.functional as F


def load_data(path):
    data = pd.read_csv(path, sep='\t', header=None, names=['source', 'relation', 'target'])
    edge_index = torch.tensor(data[['source', 'target']].to_numpy().T, dtype=torch.long)
    
    unique_nodes, new_indices = torch.unique(edge_index.flatten(), return_inverse=True)
    edge_index = new_indices.reshape(edge_index.shape)

    # Calculating the number of unique nodes
    num_nodes = unique_nodes.size(0)
    return Data(edge_index=edge_index, num_nodes=num_nodes)

# Data Loading
train_data = load_data('datasets_knowledge_embedding/WN18RR/original/train.txt')
valid_data = load_data('datasets_knowledge_embedding/WN18RR/original/valid.txt')

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, dropout_rate=0.8):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=in_channels)
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.embedding(x)
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x = self.conv2(x, edge_index)
        return x

class LinkPrediction(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(LinkPrediction, self).__init__()
        self.encoder = Encoder(in_channels, out_channels, num_nodes)  
        self.decoder = torch.nn.Linear(out_channels, 1)  

    def forward(self, edge_index):
        x = torch.arange(0, self.encoder.embedding.num_embeddings, device=edge_index.device)
        z = self.encoder(x, edge_index)
        pos_pred = self.decoder(z[edge_index[0]] * z[edge_index[1]]) 
        return torch.sigmoid(pos_pred.squeeze())


# Parameters for the model
in_channels = 16  # Input feature dimension
out_channels = 16  # Embedding dimension

model = LinkPrediction(in_channels, out_channels, num_nodes=train_data.num_nodes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()

    # Generating negative samples
    neg_edge_index = negative_sampling(edge_index=train_data.edge_index,
                                       num_nodes=train_data.num_nodes,
                                       num_neg_samples=train_data.edge_index.size(1),
                                       method='sparse')
    # Forward pass
    pos_out = model(train_data.edge_index)
    neg_out = model(neg_edge_index)

    # True labels: 1s for positive samples, 0s for negative samples
    pos_label = torch.ones(pos_out.size(0), dtype=torch.float, device=pos_out.device)
    neg_label = torch.zeros(neg_out.size(0), dtype=torch.float, device=neg_out.device)

    # Concatenate outputs and labels
    out = torch.cat([pos_out, neg_out], dim=0)
    labels = torch.cat([pos_label, neg_label], dim=0)

    # Calculate loss
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    predicted_labels = out > 0.5  # Binary classification threshold
    accuracy = (predicted_labels == labels).float().mean().item()

    return loss.item(), accuracy

def validate(model, data):
    model.eval()
    with torch.no_grad():
        # Forward pass on the validation data
        pos_out = model(data.edge_index)
        neg_edge_index = negative_sampling(edge_index=data.edge_index,
                                           num_nodes=data.num_nodes,
                                           num_neg_samples=data.edge_index.size(1),
                                           method='sparse')
        neg_out = model(neg_edge_index)

        # Concatenate outputs and true labels
        out = torch.cat([pos_out, neg_out], dim=0)
        labels = torch.cat([torch.ones(pos_out.size(0), device=pos_out.device), 
                            torch.zeros(neg_out.size(0), device=neg_out.device)], dim=0)

        # Calculate accuracy
        predicted_labels = out > 0.5
        accuracy = (predicted_labels == labels).float().mean().item()
    return accuracy

best_val_accuracy = 0
patience = 50 # This is a level to stop the training when the model is overfitting.
trigger_times = 0
train_accuracies = []
val_accuracies = []

for epoch in range(1, 101):  
    loss, train_accuracy = train()  
    val_accuracy = validate(model, valid_data)  
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Early stopping logic
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping! Epoch {epoch}")
            break

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()