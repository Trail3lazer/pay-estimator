import torch.nn as nn
import torch

#https://jonnylaw.rocks/posts/2021-08-04-entity-embeddings/
class EntityEmbedding(nn.Module):
    def __init__(self, 
                 cats: dict[str,int], 
                 conts_count: int,  
                 emb_sizes: dict[str,int], 
                 hidden_layer_dim = 100):
        super().__init__()
        self.emb_module = nn.ModuleList(nn.Embedding(cats[k],emb_sizes[k]) for k in cats)
        total_emb_size = sum(cats.values())
        self.l1 = nn.Linear(total_emb_size, hidden_layer_dim)
        self.l2 = nn.Linear(conts_count, hidden_layer_dim)
        self.relu = nn.ReLU()
        out_count = len(cats)+conts_count
        self.out = nn.Linear((2 * hidden_layer_dim), out_features=out_count)

    def forward(self, cat, cont):
        x_cat = [emb(cat[:, i]) for i, emb in enumerate(self.emb_module)]
        x_cat = torch.cat(x_cat, dim=1)
        x_cat = self.l1(x_cat)
        x_cont = self.l2(cont)
        x = torch.cat([x_cont, x_cat.squeeze()], dim=1)
        x = self.relu(x)
        return self.out(x)  