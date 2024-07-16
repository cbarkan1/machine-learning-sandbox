"""
Learning how to compute the derivative of a neural network's output
with respect to it's input. This is needed for PINNs.


Notes:

I was surprised that torch.autograd.grad() requries model output in 
addition to model input (this seems strange because in math, computing
derivative just requires the input). The reason is that the output object
records the computational graph that is needed to compute the derivative!

This raises the question: what if output and/or input are modified after
the model is run, so that output does not faithfully correspond to input?
The examples below explore this question.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(2, 1)
    
    def forward(self, x):
        x = F.relu(x)
        output = self.fc(x)
        return output


model = SimpleNet()
true_derivative = model.state_dict()['fc.weight']


# input tensor where requires_grad=True informs model(x) to...?
x = torch.tensor([[1.0, 2.0], [-3.0, -4.0]], requires_grad=True)


# EXAMPLE 1: Proper gradient computation:

output = model(x)

# I'm confused about the purpose of grad_outputs=torch.ones_like(output)
# It has something to do with linear combos between batches and/or inputs
grad = torch.autograd.grad(outputs=output, inputs=x,
                           grad_outputs=torch.ones_like(output))[0]

print("\nExample 1: Proper gradient computation")
print(f"Input: {x}")
print(f"Output: {output}")
print(f"Autograd derivative: {grad}")
print(f"True derivative: {true_derivative}\n")


# EXAMPLE 2: Modifying output after running model:

output = model(x)

# Here, output2 gets a new computational graph that includes
# the multiplication by 2! So, when torch.autograd.grad() is
# called, it knows that the derivative gets doubled!
output2 = output * 2

grad = torch.autograd.grad(outputs=output2, inputs=x,
                           grad_outputs=torch.ones_like(output2))[0]
print("\nExample 2: Multiplying output * 2")
print(f"Input: {x}")
print(f"Output: {output2}")
print(f"Autograd derivative: {grad}")
print(f"True derivative: {true_derivative}")


# Example 3: Modifying input after running model:

print("\nExample 3: Modifying input after run (throws ERROR!)\n")

output = model(x)

# This will result in an error in torch.autograd.grad(), because
# pytorch knows that x2 was nut used to compute output!
x2 = x + 1

grad = torch.autograd.grad(outputs=output, inputs=x2,
                           grad_outputs=torch.ones_like(output))[0]
