import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
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

# Create model object
model = Net()

# Load saved weights into model
model.load_state_dict(torch.load('model_weights1.pth'))


# Set the model to evaluation mode
# -- Disables dropout
# -- Disables Batch Normalization Updates (not exactly sure what this means)
# -- Ensures deterministic outputs
model.eval()

# Load testing data
transform = transforms.Compose([transforms.ToTensor()])
testset = datasets.FashionMNIST('~/data', train=False, transform=transform)

# Select an image
index = 15

# Tell pytorch not to compute gradients (which would just be a waste
# of time)
with torch.no_grad():
    image = testset[index][0].reshape(1, 1, 28, 28)
    true_class = testset[index][1]
    true_label = labels_dict[true_class]
    
    # Now, output is just logits with no grad_fn because we're 
    # using torch.no_grad()
    output = model(image)

    # the 1 is the axis along which to take max
    max_value, max_index = torch.max(output, 1)
    predicted_class = max_index.item()
    predicted_label = labels_dict[predicted_class]

    print(f'Prediction: {predicted_label} ({predicted_class})')
    print(f'Truth: {true_label} ({true_class})')


plt.imshow(testset[index][0][0, :, :], cmap='gray_r')
plt.title(labels_dict[testset[index][1]] + f' ({testset[index][1]})')
plt.show()
