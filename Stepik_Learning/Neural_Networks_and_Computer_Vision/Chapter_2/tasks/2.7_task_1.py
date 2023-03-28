import torch


tensor = torch.tensor(
    [
        [5, 10],
        [1, 2]
    ],
    requires_grad=True,
    dtype=torch.float
)

# torch.log() returns a new tensor with the natural logarithm of the elements of input.
test_tensor = torch.log(torch.log(tensor + 7))
print(test_tensor)

# torch.prod() перемножает все значения функции
function = torch.prod(test_tensor)
function.backward()

print(tensor.grad)
