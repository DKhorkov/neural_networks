import torch

tensor = torch.tensor(
    [
        [1, 3],
        [1, 5]
    ],
    requires_grad=True,
    dtype=torch.float
)

function = (tensor ** 2).prod()
print(function)  # Перемножаем все значения и получаем результат
function.backward()
print(tensor.grad)  # произведение tensor (2 * 225), а потом делим это значение на каждый элемент тензора в рамках производной
