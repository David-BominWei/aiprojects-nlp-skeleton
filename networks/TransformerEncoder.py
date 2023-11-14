import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class MainNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self, embs_npa, device, freeze_embeddings=True):
        super().__init__()
        # might need super(BaseNetwork, self).__init__()
        self.vocab_size = embs_npa.shape[0]
        # print('VOCAB SIZE', self.vocab_size)
        self.embedding_dim = embs_npa.shape[1]
        # print('EMBEDDING_DIM ', self.embedding_dim)
        self.device = device

        # freeze embeddings layer
        if freeze_embeddings:
            print('Freezing embeddings layer')


        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embs_npa).float(), 
            freeze=freeze_embeddings
        )

        # self.trans_encoder = nn.TransformerEncoder()
        
        # mid_dim = 134 * self.embedding_dim  # just for testing and trying to get things to work. 134 is max_seq_length
        # print('mid_dim ', mid_dim)  # number of samples 50
        self.fc1 = nn.Linear(self.embedding_dim, 50)
        self.fc2 = nn.Linear(50, 1)  # last layer needs to have output dim 1
        # [32, 134] dim
        # flatten in forward
        self.fc3 = nn.Linear(134, 1)

        # f1 score instead of accuracy

        self.sigmoid = nn.Sigmoid()
        self.relu = torch.nn.functional.relu


    def forward(self, input_ids): #, seq_length):
        '''
        input_ids (tensor): the input to the model
        '''

        # x = input_ids.to(self.device)  # moved to gpu in training loop
        # input_ids are ints
        print('input shape ', input_ids.shape)  # [32,134]
        embeds = self.embedding_layer(input_ids)
        print('EMBEDS output SHAPE', embeds.shape)  # [32, 134, 50]  # dims of embeddings 50d
        print(self.vocab_size, 'vocab')
        print(self.embedding_dim, 'embedding dim')
        # print()
        # sys.exit()
        # .reshape(-1. 134 * 50)
        # embeds = embeds.reshape(-1, 50)

        # x = self.fc1(x.to(torch.float32))  # original code
        # embeds are floats
        x = self.fc1(embeds)
        x = self.fc2(x)
        x = x.flatten(-2, -1)  # or squeeze
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
        # return x.reshape(-1, 1)  # original code

if __name__ == "__main__":
    from GetEmbeddings import getEmbeddings
    from StartingDataset import StartingDataset

    vocab_npa, embs_npa = getEmbeddings("glove.6B.50d.txt", '<pad>', '<unk>')
    
    val_dataset = StartingDataset("dev.csv", vocab_npa, '<pad>', '<unk>')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
    tempdata = next(iter(val_dataloader))[0]
    model = MainNetwork(embs_npa,"cpu")
    