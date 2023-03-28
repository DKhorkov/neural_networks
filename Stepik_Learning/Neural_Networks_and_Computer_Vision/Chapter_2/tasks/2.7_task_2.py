import torch

tensor = torch.tensor(
    [
        [5, 10],
        [1, 2]
    ],
    requires_grad=True,
    dtype=torch.float
)

ALPHA = 0.001  # Learning rate

for _ in range(500):
    function = torch.prod(torch.log(torch.log_(tensor + 7)))
    function.backward()
    tensor.data -= ALPHA * tensor.grad
    tensor.grad.zero_()

print(tensor)
