---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.7
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .code execution_count="101"}
``` {.python}
# %load data.py
import os
import torch

class Corpus(object):
    def __init__(self, path, batch_size, max_sql):
        self.vocabulary = []
        self.word_id = {}
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.dset_flag = "train"
        
        ## max_sql means the maximum sequence length
        self.max_sql = max_sql
        self.batch_size = batch_size
        print("size of train set: ",self.train.size(0))
        print("size of valid set: ",self.valid.size(0))
        self.train_batch_num = self.train.size(0) // self.batch_size["train"]
        self.valid_batch_num = self.valid.size(0) // self.batch_size["valid"]
        self.train = self.train.narrow(0, 0, self.batch_size["train"] * self.train_batch_num)
        self.valid = self.valid.narrow(0, 0, self.batch_size["valid"] * self.valid_batch_num)
        self.train = self.train.view(self.batch_size["train"], -1).t().contiguous()
        self.valid = self.valid.view(self.batch_size["valid"], -1).t().contiguous()

    def set_train(self):
        self.dset_flag = "train"
        self.train_si = 0

    def set_valid(self):
        self.dset_flag = "valid"
        self.valid_si = 0

    def tokenize(self, file_name):
        file_lines = open(file_name, 'r').readlines()
        num_of_words = 0
        for line in file_lines:
            words = line.split() + ['<eos>']
            num_of_words += len(words)
            for word in words:
                if word not in self.word_id:
                    self.word_id[word] = len(self.vocabulary)
                    self.vocabulary.append(word)
        file_tokens = torch.LongTensor(num_of_words)
        token_id = 0
        for line in file_lines:
            words = line.split() + ['<eos>']
            for word in words:
                file_tokens[token_id] = self.word_id[word]
                token_id += 1
        return file_tokens

    def get_batch(self):
        ## train_si and valid_si indicates the index of the start point of the current mini-batch
        if self.dset_flag == "train":
            start_index = self.train_si
            seq_len = min(self.max_sql, self.train.size(0)-self.train_si-1)
            data_loader = self.train
            self.train_si = self.train_si + seq_len
        else:
            start_index = self.valid_si
            seq_len = min(self.max_sql, self.valid.size(0)-self.valid_si-1)
            data_loader = self.valid
            self.valid_si = self.valid_si + seq_len
        data = data_loader[start_index:start_index+seq_len, :]
        target = data_loader[start_index+1:start_index+seq_len+1, :].view(-1)

        ## end_flag indicates whether a epoch (train or valid epoch) has been ended
        if self.dset_flag == "train" and self.train_si+1 == self.train.size(0):
            end_flag = True
            self.train_si = 0
        elif self.dset_flag == "valid" and self.valid_si+1 == self.valid.size(0):
            end_flag = True
            self.valid_si = 0
        else:
            end_flag = False
        return data, target, end_flag
```
:::

::: {.cell .code execution_count="102"}
``` {.python}
# %load model.py
import torch
import torch.nn as nn
from torch.nn import functional as F

class LMModel(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer. 
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding. 
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, ninput, nhid, nlayers):
        super(LMModel, self).__init__() 
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, ninput)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you RNN model here. You can add additional parameters to the function.
        self.ih2h = nn.Linear(ninput+nhid, nhid)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.h2h = nn.Linear(2*nhid,nhid)
        self.rnn = nn.RNN(ninput, nhid, nlayers)
        ########################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
    
    def my_RNNcell(self, embeddings, hidden):
        i_cat_h = torch.cat((embeddings, hidden), 2)
        f = self.ih2h(i_cat_h)
        g = self.tanh(f+hidden)
        hidden = self.sigm(f).mul(g)
        output = hidden
        return output, hidden
    
    def my_Encoder(self, embeddings, hidden, Bidirec=False):
        sql = embeddings.shape[0]
        bs = embeddings.shape[1]
        nhid = embeddings.shape[2]
        output = torch.zeros(sql, bs, nhid).to(device)
        rev_output = output
        rev_hidden = hidden
        for i in range(sql-1):
            output[i,:,:], hidden = self.my_RNNcell(embeddings[i].unsqueeze(0), hidden)
            if Bidirec == True:
                rev_output[-1-i,:,:], rev_hidden = self.my_RNNcell(embeddings[-1-i].unsqueeze(0), rev_hidden)
        if Bidirec == True:
            output = self.h2h(torch.cat((output,rev_output),2))
        return output, hidden
        
    def my_AttDecoder(self, encoder_out, last_hidden, embeddings, ndirec=1):
        attout = torch.zeros(encoder_out.size()).to(device)
        decoder_input = torch.zeros(last_hidden.size()).to(device)
        encoder_out = encoder_out.permute(1, 0, 2)
        bs = encoder_out.shape[0]
        sql = encoder_out.shape[1]
        hidden = last_hidden.to(device).detach()
        for i in range(sql-1):
            output, hidden = self.my_RNNcell(decoder_input, hidden)
            hidden = hidden.permute(1,2,0)
            attscore = torch.bmm(encoder_out, hidden)
            soft = F.softmax(attscore, dim=1)
            attdist = encoder_out*soft
            context = attdist.sum(axis=1)
            cont_hid = torch.cat((context,hidden.squeeze(2)),1)
            attout[i,:,:] = self.h2h(cont_hid)
            decoder_input = attout[i,:,:].unsqueeze(0)
            hidden = hidden.permute(2,0,1)
        return attout

    def forward(self, input, hidden, Bidirec=False):
        embeddings = self.drop(self.encoder(input))

        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        rev_hidden = hidden
        output, hidden = self.my_Encoder(embeddings, hidden, Bidirec)
            
        output = self.my_AttDecoder(output, hidden, embeddings)

#         output,hidden = self.rnn(embeddings, hidden)   
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        #decoded = self.h2o(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
```
:::

::: {.cell .code execution_count="109" scrolled="true"}
``` {.python}
# %load main.py
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

import data
# import model
import os
import os.path as osp

import time
print('gpu:',torch.cuda.is_available())

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=40, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')

args = parser.parse_args(args=[])

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
# use_gpu = False

# if use_gpu:
#     torch.cuda.set_device(args.gpu_id)
#     device = torch.device(args.gpu_id)
# else:
#     device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
# data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)
data_loader = Corpus("../data/ptb", batch_size, args.max_sql)
traindata = data_loader.train.to(device)
validdata = data_loader.valid.to(device)
epochs = args.epochs
sql = args.max_sql
        
# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)
n_input = 500
n_hid = 500
n_layers = 1
n_allchar = len(data_loader.vocabulary)

model = LMModel(nvoc=n_allchar, ninput=n_input, nhid=n_hid, nlayers=n_layers).to(device)

########################################

lr = 2.0484e-4
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9,0.999), lr=lr, weight_decay=0.0001)
# optimizer = torch.optim.Adam(model.parameters(), betas=(0.9,0.999), lr=lr)
criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.931)  
#epoch100:0.9310 epoch200:0.9649 lr:2.0484e-4
#mymodel 0.9489
#attention

# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.
validloss = []
def evaluate():
    data_loader.set_valid()
    hidden = torch.zeros(n_layers,batch_size['valid'],n_hid).to(device)

    for i in range(0,validdata.size(0)-sql,sql):
        hidden = hidden.detach()
        inputs, label, flag = data_loader.get_batch()
        output, hidden = model.forward(inputs.to(device), hidden, Bidirec=True)

        loss = criterion(output.view(-1,output.shape[2]),label.to(device))

    validloss.append(loss.item())
    print('valid loss:',loss.item(),'pp:',torch.exp(loss).item())
#     pass
########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
trainloss = []
def train():    
    data_loader.set_train()
    optimizer.zero_grad()
    hidden = torch.zeros(n_layers,batch_size['train'],n_hid).to(device)
    
    for i in range(0,traindata.size(0)-sql,sql):
        hidden = hidden.detach()
        inputs, label, flag = data_loader.get_batch()
        optimizer.zero_grad()
        output, hidden = model.forward(inputs.to(device), hidden, Bidirec=True)

        loss = criterion(output.view(-1,output.shape[2]),label.to(device))
        nn.utils.clip_grad_norm_(model.parameters(),0.5)
        loss.backward()
        optimizer.step()
        
    trainloss.append(loss.item())
    print('train loss:',loss.item(),'pp:',torch.exp(loss).item())

#     pass 
########################################


# Loop over epochs.
for epoch in range(1, args.epochs+1):
    start = time.time()
    print('epochs:',epoch)
    train()
    evaluate()
    scheduler.step()
    end = time.time()
    print('time:',end-start,'s')
    print('*'*100)
```

::: {.output .stream .stdout}
    gpu: True
    size of train set:  929589
    size of valid set:  73760
    epochs: 1
    train loss: 6.582287788391113 pp: 722.189697265625
    valid loss: 6.515632629394531 pp: 675.6212768554688
    time: 57.90779733657837 s
    ****************************************************************************************************
    epochs: 2
    train loss: 6.474143981933594 pp: 648.1641235351562
    valid loss: 6.437375068664551 pp: 624.7646484375
    time: 58.959705114364624 s
    ****************************************************************************************************
    epochs: 3
    train loss: 6.388561725616455 pp: 595.0001831054688
    valid loss: 6.3484930992126465 pp: 571.6306762695312
    time: 60.43656516075134 s
    ****************************************************************************************************
    epochs: 4
    train loss: 6.098512172698975 pp: 445.1949157714844
    valid loss: 6.104654788970947 pp: 447.93798828125
    time: 60.22092890739441 s
    ****************************************************************************************************
    epochs: 5
    train loss: 5.5479817390441895 pp: 256.71893310546875
    valid loss: 5.585790157318115 pp: 266.6108703613281
    time: 59.524882078170776 s
    ****************************************************************************************************
    epochs: 6
    train loss: 5.182116985321045 pp: 178.05935668945312
    valid loss: 5.122523307800293 pp: 167.7581329345703
    time: 59.49336051940918 s
    ****************************************************************************************************
    epochs: 7
    train loss: 4.342173099517822 pp: 76.8744125366211
    valid loss: 4.286747932434082 pp: 72.72956085205078
    time: 53.321853160858154 s
    ****************************************************************************************************
    epochs: 8
    train loss: 3.7529592514038086 pp: 42.647098541259766
    valid loss: 3.78110933303833 pp: 43.86467361450195
    time: 53.481606245040894 s
    ****************************************************************************************************
    epochs: 9
    train loss: 5.778512954711914 pp: 323.2781066894531
    valid loss: 6.055977821350098 pp: 426.6558532714844
    time: 55.031227827072144 s
    ****************************************************************************************************
    epochs: 10
    train loss: 3.0672810077667236 pp: 21.483409881591797
    valid loss: 3.176455020904541 pp: 23.961660385131836
    time: 55.02500128746033 s
    ****************************************************************************************************
    epochs: 11
    train loss: 3.298074722290039 pp: 27.060489654541016
    valid loss: 3.4137799739837646 pp: 30.379859924316406
    time: 55.57027983665466 s
    ****************************************************************************************************
    epochs: 12
    train loss: 2.796322822570801 pp: 16.384286880493164
    valid loss: 3.1230409145355225 pp: 22.715349197387695
    time: 55.022358417510986 s
    ****************************************************************************************************
    epochs: 13
    train loss: 2.252197504043579 pp: 9.508607864379883
    valid loss: 2.64017391204834 pp: 14.015641212463379
    time: 56.25506806373596 s
    ****************************************************************************************************
    epochs: 14
    train loss: 2.322044849395752 pp: 10.196503639221191
    valid loss: 2.5473039150238037 pp: 12.772621154785156
    time: 56.67420959472656 s
    ****************************************************************************************************
    epochs: 15
    train loss: 2.1030166149139404 pp: 8.190841674804688
    valid loss: 2.53141713142395 pp: 12.571308135986328
    time: 56.8003466129303 s
    ****************************************************************************************************
    epochs: 16
    train loss: 1.9843589067459106 pp: 7.2743821144104
    valid loss: 2.3147525787353516 pp: 10.122417449951172
    time: 56.63980722427368 s
    ****************************************************************************************************
    epochs: 17
    train loss: 2.079169750213623 pp: 7.997825622558594
    valid loss: 2.3702592849731445 pp: 10.700166702270508
    time: 56.59852075576782 s
    ****************************************************************************************************
    epochs: 18
    train loss: 2.064903497695923 pp: 7.8845367431640625
    valid loss: 2.268177032470703 pp: 9.661771774291992
    time: 56.34600329399109 s
    ****************************************************************************************************
    epochs: 19
    train loss: 1.8105838298797607 pp: 6.114016056060791
    valid loss: 2.0235214233398438 pp: 7.56491756439209
    time: 56.83538556098938 s
    ****************************************************************************************************
    epochs: 20
    train loss: 1.8197401762008667 pp: 6.170255184173584
    valid loss: 2.0324363708496094 pp: 7.632659435272217
    time: 58.26684284210205 s
    ****************************************************************************************************
    epochs: 21
    train loss: 2.061676025390625 pp: 7.859130859375
    valid loss: 2.2804272174835205 pp: 9.780858039855957
    time: 58.170896768569946 s
    ****************************************************************************************************
    epochs: 22
    train loss: 1.678666591644287 pp: 5.358406066894531
    valid loss: 1.8882943391799927 pp: 6.608087539672852
    time: 58.3676598072052 s
    ****************************************************************************************************
    epochs: 23
    train loss: 1.5818732976913452 pp: 4.864058971405029
    valid loss: 1.8619568347930908 pp: 6.436319351196289
    time: 56.682018995285034 s
    ****************************************************************************************************
    epochs: 24
    train loss: 4.910004138946533 pp: 135.63998413085938
    valid loss: 4.860138416290283 pp: 129.0420684814453
    time: 53.58151960372925 s
    ****************************************************************************************************
    epochs: 25
    train loss: 1.656494140625 pp: 5.240904808044434
    valid loss: 1.9194942712783813 pp: 6.817509651184082
    time: 53.91724753379822 s
    ****************************************************************************************************
    epochs: 26
    train loss: 1.6080676317214966 pp: 4.993153095245361
    valid loss: 1.8239768743515015 pp: 6.196451663970947
    time: 53.853856325149536 s
    ****************************************************************************************************
    epochs: 27
    train loss: 1.6142977476119995 pp: 5.02435827255249
    valid loss: 1.8492189645767212 pp: 6.354854106903076
    time: 53.86477851867676 s
    ****************************************************************************************************
    epochs: 28
    train loss: 1.5414944887161255 pp: 4.671566486358643
    valid loss: 1.7852739095687866 pp: 5.961212635040283
    time: 53.81159162521362 s
    ****************************************************************************************************
    epochs: 29
    train loss: 1.6186715364456177 pp: 5.046381950378418
    valid loss: 1.8861091136932373 pp: 6.593663215637207
    time: 53.22795081138611 s
    ****************************************************************************************************
    epochs: 30
    train loss: 1.4673740863800049 pp: 4.33782958984375
    valid loss: 1.7391395568847656 pp: 5.692443370819092
    time: 50.59655165672302 s
    ****************************************************************************************************
    epochs: 31
    train loss: 1.4874513149261475 pp: 4.4258012771606445
    valid loss: 1.809637427330017 pp: 6.108232021331787
    time: 47.46093463897705 s
    ****************************************************************************************************
    epochs: 32
    train loss: 1.449354648590088 pp: 4.260364532470703
    valid loss: 1.7314151525497437 pp: 5.648641586303711
    time: 47.58171248435974 s
    ****************************************************************************************************
    epochs: 33
    train loss: 1.4100261926651 pp: 4.096062660217285
    valid loss: 1.7182399034500122 pp: 5.574707508087158
    time: 47.49343514442444 s
    ****************************************************************************************************
    epochs: 34
    train loss: 1.3911385536193848 pp: 4.019423961639404
    valid loss: 1.5948576927185059 pp: 4.9276275634765625
    time: 47.87814688682556 s
    ****************************************************************************************************
    epochs: 35
:::

::: {.output .stream .stdout}
    train loss: 1.393028736114502 pp: 4.027028560638428
    valid loss: 1.8712390661239624 pp: 6.496340751647949
    time: 48.325212717056274 s
    ****************************************************************************************************
    epochs: 36
    train loss: 1.393301248550415 pp: 4.028126239776611
    valid loss: 1.72758948802948 pp: 5.627073287963867
    time: 49.76218056678772 s
    ****************************************************************************************************
    epochs: 37
    train loss: 1.3249866962432861 pp: 3.7621355056762695
    valid loss: 1.6760088205337524 pp: 5.344183921813965
    time: 48.40395498275757 s
    ****************************************************************************************************
    epochs: 38
    train loss: 1.3421120643615723 pp: 3.827117919921875
    valid loss: 1.6336413621902466 pp: 5.122493743896484
    time: 49.063406467437744 s
    ****************************************************************************************************
    epochs: 39
    train loss: 1.3098570108413696 pp: 3.705643653869629
    valid loss: 1.7184817790985107 pp: 5.576056003570557
    time: 48.87851357460022 s
    ****************************************************************************************************
    epochs: 40
    train loss: 1.327858567237854 pp: 3.7729554176330566
    valid loss: 1.7330718040466309 pp: 5.658007621765137
    time: 48.290929079055786 s
    ****************************************************************************************************
    epochs: 41
    train loss: 1.3293209075927734 pp: 3.7784764766693115
    valid loss: 1.6491413116455078 pp: 5.202510356903076
    time: 47.63530230522156 s
    ****************************************************************************************************
    epochs: 42
    train loss: 1.2956173419952393 pp: 3.6532506942749023
    valid loss: 1.626129150390625 pp: 5.084156513214111
    time: 48.027363538742065 s
    ****************************************************************************************************
    epochs: 43
    train loss: 1.281883955001831 pp: 3.603422164916992
    valid loss: 1.5924339294433594 pp: 4.915698528289795
    time: 47.89043688774109 s
    ****************************************************************************************************
    epochs: 44
    train loss: 1.2757261991500854 pp: 3.581301212310791
    valid loss: 1.5148100852966309 pp: 4.548557281494141
    time: 48.966623067855835 s
    ****************************************************************************************************
    epochs: 45
    train loss: 1.242346167564392 pp: 3.4637303352355957
    valid loss: 1.684099793434143 pp: 5.387598991394043
    time: 48.305251598358154 s
    ****************************************************************************************************
    epochs: 46
    train loss: 1.246902346611023 pp: 3.4795477390289307
    valid loss: 1.537783145904541 pp: 4.654261112213135
    time: 48.86695981025696 s
    ****************************************************************************************************
    epochs: 47
    train loss: 1.2637848854064941 pp: 3.538789987564087
    valid loss: 1.5546149015426636 pp: 4.7332634925842285
    time: 50.104448556900024 s
    ****************************************************************************************************
    epochs: 48
    train loss: 1.2334797382354736 pp: 3.4331552982330322
    valid loss: 1.5422954559326172 pp: 4.675309658050537
    time: 49.05956315994263 s
    ****************************************************************************************************
    epochs: 49
    train loss: 1.2421526908874512 pp: 3.4630603790283203
    valid loss: 1.6035362482070923 pp: 4.970578670501709
    time: 49.381513833999634 s
    ****************************************************************************************************
    epochs: 50
    train loss: 1.2368639707565308 pp: 3.444793701171875
    valid loss: 1.5103939771652222 pp: 4.528514385223389
    time: 49.164655447006226 s
    ****************************************************************************************************
    epochs: 51
    train loss: 1.222335696220398 pp: 3.395108461380005
    valid loss: 1.5955437421798706 pp: 4.931009292602539
    time: 48.89102077484131 s
    ****************************************************************************************************
    epochs: 52
    train loss: 1.2375491857528687 pp: 3.4471547603607178
    valid loss: 1.6026630401611328 pp: 4.966240406036377
    time: 48.95278882980347 s
    ****************************************************************************************************
    epochs: 53
    train loss: 1.2241384983062744 pp: 3.4012343883514404
    valid loss: 1.6248918771743774 pp: 5.077869892120361
    time: 53.30335235595703 s
    ****************************************************************************************************
    epochs: 54
    train loss: 1.2033138275146484 pp: 3.3311374187469482
    valid loss: 1.580975890159607 pp: 4.859695911407471
    time: 54.22452521324158 s
    ****************************************************************************************************
    epochs: 55
    train loss: 1.192671537399292 pp: 3.29587459564209
    valid loss: 1.5825673341751099 pp: 4.867435932159424
    time: 54.14854693412781 s
    ****************************************************************************************************
    epochs: 56
    train loss: 1.1746797561645508 pp: 3.2371063232421875
    valid loss: 1.557308554649353 pp: 4.746030330657959
    time: 54.3461594581604 s
    ****************************************************************************************************
    epochs: 57
    train loss: 1.159105896949768 pp: 3.187082529067993
    valid loss: 1.6496341228485107 pp: 5.205075263977051
    time: 54.2367889881134 s
    ****************************************************************************************************
    epochs: 58
    train loss: 1.1849396228790283 pp: 3.270489454269409
    valid loss: 1.6380923986434937 pp: 5.145345211029053
    time: 54.17982864379883 s
    ****************************************************************************************************
    epochs: 59
    train loss: 1.1742132902145386 pp: 3.2355964183807373
    valid loss: 1.5945336818695068 pp: 4.926031112670898
    time: 53.78618502616882 s
    ****************************************************************************************************
    epochs: 60
    train loss: 1.1688461303710938 pp: 3.2182769775390625
    valid loss: 1.611738920211792 pp: 5.0115180015563965
    time: 53.52145290374756 s
    ****************************************************************************************************
    epochs: 61
    train loss: 1.1767497062683105 pp: 3.2438137531280518
    valid loss: 1.4422705173492432 pp: 4.230289936065674
    time: 53.72656440734863 s
    ****************************************************************************************************
    epochs: 62
    train loss: 1.1731040477752686 pp: 3.2320094108581543
    valid loss: 1.4300779104232788 pp: 4.179025173187256
    time: 53.65708351135254 s
    ****************************************************************************************************
    epochs: 63
    train loss: 1.1766175031661987 pp: 3.243385076522827
    valid loss: 1.5126168727874756 pp: 4.5385918617248535
    time: 53.74785900115967 s
    ****************************************************************************************************
    epochs: 64
    train loss: 1.1761215925216675 pp: 3.241776943206787
    valid loss: 1.5688940286636353 pp: 4.801335334777832
    time: 50.17041802406311 s
    ****************************************************************************************************
    epochs: 65
    train loss: 1.141945719718933 pp: 3.1328580379486084
    valid loss: 1.5383731126785278 pp: 4.657007694244385
    time: 48.368446350097656 s
    ****************************************************************************************************
    epochs: 66
    train loss: 1.172635555267334 pp: 3.2304954528808594
    valid loss: 1.5813106298446655 pp: 4.86132287979126
    time: 53.434295415878296 s
    ****************************************************************************************************
    epochs: 67
    train loss: 1.169973373413086 pp: 3.221906900405884
    valid loss: 1.6178046464920044 pp: 5.042008876800537
    time: 49.735357999801636 s
    ****************************************************************************************************
    epochs: 68
    train loss: 1.1676452159881592 pp: 3.214414358139038
    valid loss: 1.4591758251190186 pp: 4.302412033081055
    time: 51.232825756073 s
    ****************************************************************************************************
    epochs: 69
:::

::: {.output .stream .stdout}
    train loss: 1.1745867729187012 pp: 3.236804962158203
    valid loss: 1.4609370231628418 pp: 4.309996128082275
    time: 52.983487606048584 s
    ****************************************************************************************************
    epochs: 70
    train loss: 1.1785106658935547 pp: 3.2495310306549072
    valid loss: 1.4226188659667969 pp: 4.1479692459106445
    time: 50.05758762359619 s
    ****************************************************************************************************
    epochs: 71
    train loss: 1.1970492601394653 pp: 3.3103346824645996
    valid loss: 1.4879192113876343 pp: 4.427872180938721
    time: 54.77777910232544 s
    ****************************************************************************************************
    epochs: 72
    train loss: 1.1601107120513916 pp: 3.19028639793396
    valid loss: 1.6421698331832886 pp: 5.166367530822754
    time: 50.71821427345276 s
    ****************************************************************************************************
    epochs: 73
    train loss: 1.1491312980651855 pp: 3.1554505825042725
    valid loss: 1.6234277486801147 pp: 5.070440769195557
    time: 55.39125609397888 s
    ****************************************************************************************************
    epochs: 74
    train loss: 1.1309900283813477 pp: 3.0987229347229004
    valid loss: 1.4973344802856445 pp: 4.469758987426758
    time: 50.528356075286865 s
    ****************************************************************************************************
    epochs: 75
    train loss: 1.1589878797531128 pp: 3.186706304550171
    valid loss: 1.378741979598999 pp: 3.9699041843414307
    time: 53.11100149154663 s
    ****************************************************************************************************
    epochs: 76
    train loss: 1.1599928140640259 pp: 3.1899101734161377
    valid loss: 1.516169786453247 pp: 4.554746150970459
    time: 52.22172832489014 s
    ****************************************************************************************************
    epochs: 77
    train loss: 1.144680142402649 pp: 3.1414363384246826
    valid loss: 1.5030573606491089 pp: 4.495412349700928
    time: 49.79028797149658 s
    ****************************************************************************************************
    epochs: 78
    train loss: 1.1487133502960205 pp: 3.154132127761841
    valid loss: 1.638121485710144 pp: 5.1454949378967285
    time: 49.91085481643677 s
    ****************************************************************************************************
    epochs: 79
    train loss: 1.1441636085510254 pp: 3.1398141384124756
    valid loss: 1.5382091999053955 pp: 4.65624475479126
    time: 50.823716163635254 s
    ****************************************************************************************************
    epochs: 80
    train loss: 1.1469955444335938 pp: 3.1487185955047607
    valid loss: 1.5651142597198486 pp: 4.783221244812012
    time: 51.03503751754761 s
    ****************************************************************************************************
    epochs: 81
    train loss: 1.1678462028503418 pp: 3.2150607109069824
    valid loss: 1.5459362268447876 pp: 4.6923627853393555
    time: 50.83203196525574 s
    ****************************************************************************************************
    epochs: 82
    train loss: 1.169385313987732 pp: 3.220012664794922
    valid loss: 1.372576117515564 pp: 3.9455018043518066
    time: 53.74902009963989 s
    ****************************************************************************************************
    epochs: 83
    train loss: 1.1682559251785278 pp: 3.2163782119750977
    valid loss: 1.3684520721435547 pp: 3.9292638301849365
    time: 51.69697165489197 s
    ****************************************************************************************************
    epochs: 84
    train loss: 1.1480522155761719 pp: 3.152047634124756
    valid loss: 1.5467709302902222 pp: 4.6962809562683105
    time: 51.905320167541504 s
    ****************************************************************************************************
    epochs: 85
    train loss: 1.1490381956100464 pp: 3.1551568508148193
    valid loss: 1.447617530822754 pp: 4.252970218658447
    time: 52.66736435890198 s
    ****************************************************************************************************
    epochs: 86
    train loss: 1.1361463069915771 pp: 3.1147420406341553
    valid loss: 1.5264794826507568 pp: 4.601946830749512
    time: 50.586790800094604 s
    ****************************************************************************************************
    epochs: 87
    train loss: 1.1382129192352295 pp: 3.121185541152954
    valid loss: 1.4831125736236572 pp: 4.40664005279541
    time: 55.03611779212952 s
    ****************************************************************************************************
    epochs: 88
    train loss: 1.1514567136764526 pp: 3.162796974182129
    valid loss: 1.4483263492584229 pp: 4.255985736846924
    time: 50.575573444366455 s
    ****************************************************************************************************
    epochs: 89
    train loss: 1.1381345987319946 pp: 3.120941162109375
    valid loss: 1.5223335027694702 pp: 4.582907199859619
    time: 55.32025933265686 s
    ****************************************************************************************************
    epochs: 90
    train loss: 1.1579325199127197 pp: 3.183344841003418
    valid loss: 1.490864634513855 pp: 4.440933704376221
    time: 52.0335111618042 s
    ****************************************************************************************************
    epochs: 91
    train loss: 1.1523411273956299 pp: 3.165595293045044
    valid loss: 1.5248461961746216 pp: 4.594437122344971
    time: 54.02604413032532 s
    ****************************************************************************************************
    epochs: 92
    train loss: 1.152387022972107 pp: 3.165740728378296
    valid loss: 1.508862853050232 pp: 4.521585941314697
    time: 53.58848810195923 s
    ****************************************************************************************************
    epochs: 93
    train loss: 1.1376665830612183 pp: 3.119480848312378
    valid loss: 1.5257657766342163 pp: 4.598663806915283
    time: 51.5235071182251 s
    ****************************************************************************************************
    epochs: 94
    train loss: 1.1383428573608398 pp: 3.121591091156006
    valid loss: 1.628493070602417 pp: 5.096189498901367
    time: 51.05287766456604 s
    ****************************************************************************************************
    epochs: 95
    train loss: 1.1498544216156006 pp: 3.1577329635620117
    valid loss: 1.4843896627426147 pp: 4.412271499633789
    time: 50.97371506690979 s
    ****************************************************************************************************
    epochs: 96
    train loss: 1.1472415924072266 pp: 3.1494932174682617
    valid loss: 1.5072842836380005 pp: 4.514453887939453
    time: 51.34265327453613 s
    ****************************************************************************************************
    epochs: 97
    train loss: 1.1368353366851807 pp: 3.116888999938965
    valid loss: 1.5835028886795044 pp: 4.8719916343688965
    time: 50.74285697937012 s
    ****************************************************************************************************
    epochs: 98
    train loss: 1.1420284509658813 pp: 3.133117198944092
    valid loss: 1.5303043127059937 pp: 4.619582176208496
    time: 50.50823473930359 s
    ****************************************************************************************************
    epochs: 99
    train loss: 1.1431143283843994 pp: 3.136521100997925
    valid loss: 1.492294430732727 pp: 4.4472880363464355
    time: 50.501548767089844 s
    ****************************************************************************************************
    epochs: 100
    train loss: 1.1558297872543335 pp: 3.1766583919525146
    valid loss: 1.5617034435272217 pp: 4.766934394836426
    time: 51.05300450325012 s
    ****************************************************************************************************
:::
:::

::: {.cell .code execution_count="194"}
``` {.python}
from nltk.translate.bleu_score import sentence_bleu
max_sql = 35
batch_size = {'train': 1,'valid':1}
data_loader = Corpus("../data/ptb", batch_size, max_sql)
data_loader.set_valid()
bleu_opt = 0
bleu = []
for i in range(0,len(data_loader.valid)-sql,sql):
# for i in range(5):
    inputs, label, flag = data_loader.get_batch()
    hidden = torch.zeros(n_layers,batch_size['valid'],n_hid).to(device)
    output, hidden = model.forward(inputs.to(device), hidden, Bidirec=True)

    reference_ind = torch.max(output.view(-1,output.shape[2]),1)[1].cpu().numpy().tolist()
    candidate_ind = label.cpu().numpy().tolist()
    reference = [[data_loader.vocabulary[i] for i in reference_ind]]
    candidate = [data_loader.vocabulary[i] for i in candidate_ind]
    bleuscore = sentence_bleu(reference, candidate)
    bleu.append(bleuscore)
    if bleuscore > bleu_opt:
        bleu_opt = bleuscore
        reference_opt = reference
        candidate_opt = candidate
        
print('\nSequence length: ',max_sql,'\n')
print('Avg_BLEU: %.4f' % torch.mean(torch.Tensor(bleu)),'\n')
print('='*40,'Optimal Case','='*40)
print('BLEU: %.4f'% bleu_opt)
print('\nReal text:\n'," ".join(candidate_opt))
print('\nPrediction text:\n'," ".join(reference_opt[0]))
```

::: {.output .stream .stdout}

    Sequence length:  35 

    Avg_BLEU: 0.6692 

    ======================================== Optimal Case ========================================
    BLEU: 0.9701

    Real text:
     <eos> the smaller stocks in the tokyo market 's second section also posted their biggest decline of the year <eos> the tokyo stock exchange index for the second section fell N or N N to

    Prediction text:
     <eos> the smaller stocks in the tokyo market 's second section also posted their biggest decline of the year <eos> the tokyo stock exchange index for the second section fell N or N N the
:::
:::

::: {.cell .code execution_count="29" scrolled="false"}
``` {.python}
# emb size 1000
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/ea64148df5078ef132e7edd02946ca5bae00aeac.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/b2d5f2081ffe8a5d90b15f3183cdeb82a7cb5edb.png)
:::
:::

::: {.cell .code execution_count="34"}
``` {.python}
# learning rate strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/af62d70551b839c0a35731c387aff02687ab1f24.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/dead65a088f409dfa6cc0b69e822b74420d61611.png)
:::
:::

::: {.cell .code execution_count="39"}
``` {.python}
# after learning rate strategy
# 2.0484e-4 ~ 1.6105e-7 -> gamma = 0.931
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/7efef84c5def9d7505a71b3092d9fe662d005f48.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/e7796b221c78553bfed1204531b1c0e8e032a094.png)
:::
:::

::: {.cell .code execution_count="44"}
``` {.python}
# num layer = 2
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/00499387d69633b10045963769b6588effaa418c.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/341e16454058f29b530f8cac797cdab196feda17.png)
:::
:::

::: {.cell .code execution_count="46"}
``` {.python}
# num layer = 3
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/20e681695ca273a97d0ae0cb15396f7a0174c2d8.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/95ca6b34ad1465dfc3ccb9f5f23554d1ad3cf434.png)
:::
:::

::: {.cell .code execution_count="60"}
``` {.python}
# num layer = 3, weigt decay = 0.001, lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/9b5bef79afef591c9d4ea7c730d449c9328e24bf.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/c3a1f7ea3993f929f20df4f12695df7d24c2b619.png)
:::
:::

::: {.cell .code execution_count="63"}
``` {.python}
# num layer = 3, weigt decay = 0.001, after lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/900536edbb1e371fa7608e3ba618acf9336655ce.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/2c29466c9fd59ca86c5ec41baa904f2f92ebe320.png)
:::
:::

::: {.cell .code execution_count="65"}
``` {.python}
# num layer = 1, weigt decay = 0.001, lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/f77254e626a926af9477122f6760df42addb00c0.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/7ae3e62ae1108f74abb56cf11e2dc9e5bd5953d9.png)
:::
:::

::: {.cell .code execution_count="67"}
``` {.python}
# num layer = 1, weigt decay = 0.001, after lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/836a1a964e099d17deb7287a8545b87921e364c6.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/8c80ee84b5af8436aa24a08258dc0a8f0422aa51.png)
:::
:::

::: {.cell .code execution_count="69"}
``` {.python}
# num layer = 1, weigt decay = 0.0001, after lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/3c096c98b4cd5c6cd0e2d9350271aa4c55297876.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/fa5b593e294e0fc14c1e4b483c141c70ba3add68.png)
:::
:::

::: {.cell .code execution_count="71"}
``` {.python}
# num layer = 3, weigt decay = 0.0001, after lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/002ba5ced5bd508aec3dc447bef806a9954b4ac8.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/0c240bba9e00ddd7a036dff177ee64172743a933.png)
:::
:::

::: {.cell .code execution_count="75"}
``` {.python}
# num layer = 3, weigt decay = 0.0002, after lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/a76a6d99b19511addbd743898d9bb8a20e72b786.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/089abba3c576fcf31aafe7e1b65d2e94fca6dec1.png)
:::
:::

::: {.cell .code execution_count="73"}
``` {.python}
# num layer = 3, weigt decay = 0.0005, after lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/dbd2edd828215e4c73c77bec7ce483b7c78428a1.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/6b96b1713a871f7831a09adb6652a0371aef7ecf.png)
:::
:::

::: {.cell .code execution_count="78"}
``` {.python}
# num layer = 3, weigt decay = 0.00001, after lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/928fe96b4ecf54552249e119831deea401e619ea.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/c401baafc1b3ac2ead18b2a1be660d3b6e5f22b5.png)
:::
:::

::: {.cell .code execution_count="198"}
``` {.python}
# my model, lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/564f8fbc38f875c34450c48338f8552570921cc7.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/8476f24625146adf5b06812c5cd9a6690a92124d.png)
:::
:::

::: {.cell .code execution_count="208"}
``` {.python}
# my model, weight_decay = 5e-6, after lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/98a05b72d58fcf1a1b8c0339a2e645259a219515.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/b6c87e4ac20621747f15ae4f08ab527561cfe2fd.png)
:::
:::

::: {.cell .code execution_count="151"}
``` {.python}
# my model, weight_decay = 0.0001, Bidirectional, lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/e331936263131ab39d4dfd4ca8f5d90942365402.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/f9e58cae68c92b2d04cf9b56d831750f0d3ec40d.png)
:::
:::

::: {.cell .code execution_count="167"}
``` {.python}
# my model, weight_decay = 0.0001, Bidirectional, after lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt-10), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/8db2dbaf1bed9b059bf47889701e4ee1e16dda0f.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/541b4a22231262afd428bee49b77c84ece2f9d5b.png)
:::
:::

::: {.cell .code execution_count="107"}
``` {.python}
# my model, weight_decay = 0.0001, Bidirectional, Attention, lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt-10), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/de47bfb15293667074b9eab730e1c1f7431c9352.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/8eb6af0c0569876e5d8a9f6961ab1e57149ba8ad.png)
:::
:::

::: {.cell .code execution_count="115"}
``` {.python}
# my model, weight_decay = 0.0001, Bidirectional, Attention, lr strategy
import matplotlib.pyplot as plt

plt.plot(trainloss,label = 'training')
plt.plot(validloss,label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
xt = len(trainloss)-1
yt = trainloss[xt]
plt.annotate('Loss: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = validloss[xv]
plt.annotate('Loss: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()

plt.plot(torch.exp(torch.Tensor(trainloss)),label = 'training')
plt.plot(torch.exp(torch.Tensor(validloss)),label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity')
xt = len(trainloss)-1
yt = torch.exp(torch.Tensor(trainloss))[xt]
plt.annotate('PP: {:.4f}'.format(yt), xy = (xt,yt), xytext = (xt*1.1,yt-30), arrowprops=dict(arrowstyle='->'))
xv = len(validloss)-1
yv = torch.exp(torch.Tensor(validloss))[xv]
plt.annotate('PP: {:.4f}'.format(yv), xy = (xv,yv), xytext = (xv*1.1,yv), arrowprops=dict(arrowstyle='->'))
plt.show()
```

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/74e19b6320cc36d48502841ed89e5f1aad722568.png)
:::

::: {.output .display_data}
![](vertopal_26fb8ff10a7c45d8851a1b742c075019/08b3fdc9177bbbd78e8cc78e82dfce96c225162e.png)
:::
:::
