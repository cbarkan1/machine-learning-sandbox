"""
The following blog post by Adrian Tam was used for reference:
https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# load the dataset. The first 8 columns are the patient data, and the
# last column indicates whether the patient has diabetes.
dataset = np.loadtxt('pima-indians-diabetes-data.csv', delimiter=',')

num_datapoints = dataset.shape[0]
input_dim = dataset.shape[1] - 1

# Using 3/4 of data for training and 1/4 for testing.
X_train = dataset[:num_datapoints*3//4, :-1]
y_train = dataset[:num_datapoints*3//4, -1]
X_test = dataset[num_datapoints*3//4:, :-1]
y_test = dataset[num_datapoints*3//4:, -1]


# Need to convert numpy arrays to pytorch tensors.
# y must be converted to shape nx1
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# The model object defined below stores both the model architecture,
# and the model parameters. For each parameter, its component of the
# gradient is also stored in its .grad attribute.
model = nn.Sequential(
    nn.Linear(input_dim, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)


# Forward pass of model (before training):
# Output looks like: tensor([0.9164], grad_fn=<SigmoidBackward0>)
# The 0.9164 is the numerical output of the neural network
# The grad_fn=<SigmoidBackward0> is "metadata" about computing gradients.
# <SigmoidBackward0> is just the first function in a chain of grad_fn functions
# used for back propogation.
print(model(X_train[0, :]))


# BCELoss() takes a probability as input (a number in [0,1])
# Hence, we need the sigmoid layer in our network in order to use
# this loss function.
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)


N_EPOCHS = 100
BATCH_SIZE = 100

for epoch in range(N_EPOCHS):
    for i in range(0, len(X_train), BATCH_SIZE):
        Xbatch = X_train[i:i+BATCH_SIZE]
        ybatch = y_train[i:i+BATCH_SIZE]

        # Forward pass through the model builds the computational graph
        # (i.e. creates the pointers from each grad_fn to the previous
        # grad_fn). Then, loss.backward() traces back through the 
        # computational graph to compute the gradient.
        y_pred = model(Xbatch)

        # Computes loss AND computes gradient. Gradient data is saved 
        # in the .grad attribute of each parameter. Note that
        # model.parameters is a "generator" that iterates over parameters.
        # Note: the loss is the average loss for the datapoints in the batch.
        loss = loss_fn(y_pred, ybatch)
        
        # Note: if you do print(loss) you'll see that loss has a grad_fn
        # for BCELoss. This grad_fn has a pointer to the grad_fn for y_pred,
        # which itself has a pointer to the grad_fn for the previous
        # layer, and so on.

        # Gradient must be set to zero, or else it'll be added on
        # to the previously-stored gradient.
        optimizer.zero_grad()
        
        # loss.backward() computes the gradient and saves the component 
        # for each parameter in the model.parameters() tensor.
        # Backpropogation can trace back through the network because
        # the loss object has a grad_fn t
        loss.backward()

        optimizer.step()

    print(f'Finished epoch {epoch}, latest loss {loss}')


# Check accuracy with test data
y_pred = model(X_test)
accuracy = (y_pred.round() == y_test).float().mean()
print(f"Accuracy {accuracy}")
