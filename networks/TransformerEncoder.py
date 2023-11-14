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
        
        encoder_layers = nn.TransformerEncoderLayer(50, 2, 50, True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 2)


    def forward(self, input_ids): #, seq_length):
        
        embeds = self.embedding_layer(input_ids) # shape batch(32) maxlen(134) embeddim(50)
        encode_tensor = self.transformer_encoder(embeds, ...) # TODO: finish the encode layer
        print(embeds.shape)
        
        return embeds
        

if __name__ == "__main__":
    from GetEmbeddings import getEmbeddings
    from StartingDataset import StartingDataset

    vocab_npa, embs_npa = getEmbeddings("glove.6B.50d.txt", '<pad>', '<unk>')
    
    val_dataset = StartingDataset("dev.csv", vocab_npa, '<pad>', '<unk>')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
    tempdata = next(iter(val_dataloader))[0]
    model = MainNetwork(embs_npa,"cpu")
    print(model(next(iter(val_dataloader))[0]).shape)

    