# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import plot

# Constructing the training set and the test set
training_set = pd.read_csv('dataset/ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('dataset/ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Obtaining the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# To get an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 = Liked or 0 = Not Liked -1 = Unrated
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

epoch_list = []

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    epoch_list.append(train_loss/s)  
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))

total_no_of_negative = 0
total_no_of_zero = 0
total_no_of_one = 0

viewer_no_of_negative = 0
viewer_no_of_zero = 0
viewer_no_of_one = 0

viewers = [[0 for x in range((3))] for y in range(nb_users)]
for i in range(len(training_set)):
    for j in range(len(training_set[i])):
        if(training_set[i][j] == -1):
            total_no_of_negative = total_no_of_negative + 1
            viewer_no_of_negative = viewer_no_of_negative + 1
        elif(training_set[i][j] == 0):
            total_no_of_zero = total_no_of_zero + 1
            viewer_no_of_zero = viewer_no_of_zero + 1
        else:
            total_no_of_one = total_no_of_one + 1
            viewer_no_of_one = viewer_no_of_one + 1
    viewers[i][0] = viewer_no_of_one
    viewers[i][1] = viewer_no_of_zero
    viewers[i][2] = viewer_no_of_one
    viewer_no_of_negative = 0
    viewer_no_of_zero = 0
    viewer_no_of_one = 0
        



# Plotting total number of rating irrespective of users

#Plot the graph individually
    
trace0 = go.Bar(x=['Negative Rating','Positive Rating'],
           y=[total_no_of_zero,total_no_of_one], opacity=0.6)
    
layout0 = go.Layout(title = "Rating Chart Total users vs Total movies", xaxis = dict(
        title='Ratings',
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black')
            ),
        yaxis = dict(
        title='Total number of users',
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black')
            )
        )

data0 = [trace0]
fig0 = go.Figure(data=data0, layout=layout0)
plot(fig0)



#   Plot to display ratings for first 5 users. 

trace1 = go.Bar(
    x=['User1', 'User2', 'User3', 'User4', 'User5'],
    y=[viewers[0][1], viewers[1][1], viewers[2][1], viewers[3][1], viewers[4][1]],
    name='Negative Rating'
)
trace2 = go.Bar(
    x=['User1', 'User2', 'User3', 'User4', 'User5'],
    y=[viewers[0][2], viewers[1][2], viewers[2][2], viewers[3][2], viewers[4][2]],
    name='Positive Rating'
)

data2 = [trace1, trace2]
layout2 = go.Layout(
    barmode='group'
)

fig2 = go.Figure(data=data2, layout=layout2)
plot(fig2)


