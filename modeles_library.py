import torch
import torch.nn as nn

class network(nn.Module):

  def __init__(self):
    super().__init__()

  def count_parameters(self, count_non_trainable_paramters=True):
    if count_non_trainable_paramters == True:
      return sum(p.numel() for p in self.parameters())
    else:
      return sum(p.numel() for p in model.parameters() if p.requires_grad)



class my_lstm(nn.Module):

  def __init__(self, embedding_dim, hidden_dim, target_size, num_layers=1):
    super(my_lstm, self).__init__()

    self.hidden_dim = hidden_dim
    self.embedding_dim = embedding_dim
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)

    self.hidden2output = nn.Linear(hidden_dim, target_size)

  def forward(self, seq):
    lstm_out, _ = self.lstm(seq.view(len(seq),1,-1))
    out = self.hidden2output(lstm_out[-1])
    return out.squeeze()

