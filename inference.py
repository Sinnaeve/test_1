import os
import torch

from modeles_library import my_lstm


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_lstm(embedding_dim=1, hidden_dim=1, target_size=1, num_layers=1)
    print("model:", model)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

print("studio push succes")