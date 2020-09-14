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
        
        # Call parent __init__
        super().__init__()
        
        # Save member data
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_prob = 0.1
        
        # Store embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        # Define LSTM Cell
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, self.dropout_prob, batch_first = True)   
        
        # Add dropout layer to reduce likelihood of overfitting
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Final linear fullly connected layer for output
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Initialize weights
        # Initialize bias tensor
        self.fc.bias.data.fill_(0)
        # Initialize fully connected weights with random values
        self.fc.weight.data.uniform_(-1,1)
        
    
    def forward(self, features, captions):
        
        # Remove <end> token 
        captions = captions[:, :-1]
        
        # Pass thru embedding layer
        captions = self.embedding(captions)
        
        # Concatenate the image feature vector and captions to create input tensor
        lstm_in = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        # Pass thru LSTM cells
        lstm_out, lstm_hidden = self.lstm(lstm_in)
        
        # Pass thru dropout layer
        lstm_out = self.dropout(lstm_out)
        
        # Pass thru linear fully connected output layer
        lstm_out = self.fc(lstm_out)
        
        return lstm_out
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        caption_words = []
        for k in range(max_len):
            
            # LSTM cells
            lstm_output, states = self.lstm(inputs, states)
            
            # Output layer
            lstm_output = self.fc(lstm_output.squeeze(dim=1))
            
            # Get prediction
            _, caption_word_index = torch.max(lstm_output, dim=1)
            
            # Add to index to caption 
            caption_words.append(caption_word_index.item())
            
            if (caption_word_index == 1):
                break    # Stop if <end> token found
            
            # Update for next iteration
            inputs = self.embedding(caption_word_index)
            inputs = inputs.unsqueeze(1)
            
        return caption_words
