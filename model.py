import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
        #self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        #self.sig = nn.Sigmoid()


    
    def forward(self, features, captions):
        batch_size = features.shape[0]
        #self.hidden = self.init_hidden(batch_size)

        
        #self.hidden = (torch.zeros(1, 1, hidden_dim),torch.zeros(1, 1, hidden_dim)) 
        embeds = self.word_embeddings(captions[:,:-1]) #instead of captions[:,:-1], features 
        embeds = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)
        
        # the first value returned by LSTM is all of the hidden states throughout
        # the sequence. the second is just the most recent hidden state
        # Add the extra 2nd dimension
        #inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        #hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
        lstm_out, _ = self.lstm(embeds)
        
        out = self.fc(lstm_out)
        #out = torch.nn.functional.log_softmax(tag_space, dim=1)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []

        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            tag_output = self.fc(lstm_out)

            # don't need softmax, since eventually after it argmax will return the same index
            predicted = torch.argmax(tag_output, dim=-1)

            result.append(predicted[0,0].item())
            inputs = self.word_embeddings(predicted)
        return result