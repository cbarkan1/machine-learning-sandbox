import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from time import time
from model_architecture import Net


labels_dict = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# What exactly is this "transform" object?
transform = transforms.Compose([transforms.ToTensor()])

# download=True is needed only in the first of the following lines, 
# because once the data is downloaded it doesn't need to be downloaded
# again. Even if download=True, download will occur only if the files
# aren't already present.
trainset = datasets.FashionMNIST(
    '~/data', train=True, download=True, transform=transform)
testset = datasets.FashionMNIST('~/data', train=False, transform=transform)

# Notes:
# -- Images are 1x28x28
# -- There are 60,000 images in the training dataset
# -- There are 10,000 images in the testing dataset

"""
# Plot an image
i = 59  # image index
plt.imshow(testset[i][0][0, :, :], cmap='gray_r')
plt.title(labels_dict[testset[i][1]] + f' ({testset[i][1]})')
plt.show()
"""

# Reducing size of training set to 20,000 images to speed things up:
indices = list(range(20000))
trainset = torch.utils.data.Subset(trainset, indices)


# These objects handle batching and shuffling of data
# DataLoader uses parallel processing for shuffling and batching,
# which speeds things up.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=True)

# Create the model instance
model = Net()

"""
# Run the model
# Output lists the logits, and the grad_fn of the final layer.
test_input = trainset[0][0].reshape(1, 1, 28, 28)
output = model(test_input)
print(output)
"""

# CrossEntropyLoss takes logits as inputs, so the values do not need
# to be between 0 and 1.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the model
epochs = 6
for epoch in range(epochs):
    loss_counter = 0.0
    iter_count = 0
    for inputs, labels in trainloader:
        # inputs.shape = batch_size x 1 x 28 x 28
        # labels.shape = batch_size
        # For batch_size=20, each iteration takes about .025s, which
        # is .0013s/image.
        
        # time0 = time()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_counter += loss.item()
        iter_count += 1

        # time1 = time()
        # print(iter_count, ' Batch time = ', time1-time0)

    print(f'Epoch {epoch+1} loss: {loss_counter / len(trainloader):.3f}')


# Evaluate the model
correct = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

print(f'Test accuracy: {100 * correct / 10000}%')


# Save the model parameters
torch.save(model.state_dict(), 'model_weights1.pth')
