
import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score

df=pd.read_csv('yelp_academic_dataset_review.csv',nrows=100000)
# df.head()
# print(df.shape)
df_filtered=df[df['stars'] !=3] 
# print(df_filtered.shape)

#print(df_filtered.describe().T)

text=list(df_filtered['text'])
stars=list(df_filtered['stars'])

print(type(text))
label=[]

for item in stars:
    if item>= 4:
        y=1
    else:
        y=0
    label.append(y)
label=np.array(label)
#we can get punctuation from string library
from string import punctuation
print(punctuation)

all_reviews=[]
for item in text:
  item = item.lower()
  item = "".join([ch for ch in item if ch not in punctuation])
  all_reviews.append(item)
all_text = " ".join(all_reviews)
print(all_text[0:20])
all_words = all_text.split()
print(all_words[0:10])


from collections import Counter 
# Count all the words using Counter Method
count_words = Counter(all_words)
total_words=len(all_words)
sorted_words=count_words.most_common(total_words)
#print(sorted_words[:30])

vocab_to_int={w:i+1 for i,(w,c) in enumerate(sorted_words)}
#print(vocab_to_int)


#reviews_ints = []
#for review in all_words:
  #reviews_ints.append([vocab_to_int[word] for word in all_words])

encoded_reviews=list()
for review in all_reviews:
    encoded_review=list()
    for word in review.split():
        if word not in vocab_to_int.keys():
          #if word is not available in vocab_to_int put 0 in that place
            encoded_review.append(0)
        else:
            encoded_review.append(vocab_to_int[word])

    if len(encoded_review) == 0:
        encoded_reviews.append([0])
    else:
        encoded_reviews.append(encoded_review)

reviews_len = [len(x) for x in encoded_reviews]
pd.Series(reviews_len).hist()
plt.xlabel('Words')
plt.ylabel('Count')
plt.show()

# stats about vocabulary
#print('Unique words: ', len((vocab_to_int)))  
#print()

# print tokens in first review
#print('Tokenized review: \n', encoded_reviews[:1])

def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    ## getting the correct rows x cols shape
    features = np.zeros((len(encoded_reviews), seq_length), dtype=int)
    
    ## for each review, I grab that review
    for i, row in enumerate(encoded_reviews):
      features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features

seq_length = 200

features = pad_features(encoded_reviews, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(encoded_reviews), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches 
#print(features[:30,:10])

train_x, test_x, train_y, test_y = model_selection.train_test_split(features,label, test_size=0.2, random_state=42)

## print out the shapes of your resultant feature data
print("\t\t\tFeatures Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

#train_x=np.array(train_x).astype('float')
#train_y=np.array(train_x).astype('float')
#test_x=np.array(train_x).astype('float')
#test_y=np.array(train_x).astype('float')

import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 128

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

# print('Sample input size: ', sample_x.size()) # batch_size, seq_length
# print('Sample input: \n', sample_x)
# print()
# print('Sample label size: ', sample_y.size()) # batch_size
# print('Sample label: \n', sample_y)


# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size, train_on_gpu):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if(train_on_gpu):
          hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
          hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1 # +1 for zero padding + our word tokens
output_size = 1
embedding_dim = 400 
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)


# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params

epochs = 4 

counter = 0
print_every = 100
clip=5 # gradient clipping
train_on_gpu = True
# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()

# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size, train_on_gpu)
    counter = 0
    # batch loop
    for inputs, labels in train_loader:
        counter += 1
        #print('epoce: {e}, batch: {b}'.format(e=e, b=counter))
        if (labels.shape[0] != batch_size):
            continue


        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        # zero accumulated gradients
        net.zero_grad()

        
        output, h = net(inputs, h)
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()





# Get test data loss and accuracy

# = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size, train_on_gpu)
counter=0
net.eval()

all_prediction = []
# iterate over test data
for inputs, labels in test_loader:
    counter += 1
    print('epoce: {e}, batch: {b}'.format(e=e, b=counter))
    if (labels.shape[0] != batch_size):
        continue    
    print(inputs.shape)
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # get predicted outputs
    output, h = net(inputs, h)

    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)
    

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))


