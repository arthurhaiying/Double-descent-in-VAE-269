import torch
import torch.nn as nn

class SkipGRU(nn.Module):

    def __init__(self,input_size,hidden_size,latent_size,seq_len):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.rnn_cell = nn.GRUCell(input_size,hidden_size)
        self.linear = nn.Linear(hidden_size+latent_size,hidden_size)

    def forward(self,inputs,h0,z):
        seq_len = inputs.size(1)
        ht = h0
        #print("h0 shape %s" %(h0.size(),))
        #print("z shape %s" %(z.size(),))
        # for each time step
        outputs = []
        for i in range(seq_len):
            xt = inputs[:,i]
            ht = torch.cat((ht,z),dim=-1)
            ht_hat = self.linear(ht) 
            # concatenate hidden and latent states
            ht = self.rnn_cell(xt,ht_hat)
            outputs.append(ht)

        outputs = torch.stack(outputs,dim=0)
        outputs = torch.transpose(outputs,0,1)
        #print("outputs size %s" %(outputs.size(),))
        return outputs, ht


