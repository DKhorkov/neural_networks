import torch


tensor = torch.tensor(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    requires_grad=True,  # Данный атрибут включен для дальнейшего взятия производной тензора,
    dtype=torch.float  # Only Tensors of floating point and complex dtype can require gradients
)


LR = 0.001  # Скорость обучения (learning rate) / скорость градиентного спуска
function = 10 * (tensor ** 2).sum()

"""
Возьмем производную от нашей функции, то есть 10 * производную от "x" в степени 2 = 10 * 2* "x" в степени 1

Backward - совершение операций в обратном порядке для взятия производной, то есть обратное от следующего порядка:
    1) умножить "х" на 2;
    2) производится суммирование по всем компонентам аргумента "х";
    3) умножаем результат на 10
"""
function.backward()

# Берем производную от функции, но операция происходит с тензором (самим "X"), то есть изменяем тензор
print(tensor.grad)

# Просмотр порядка выполнения операций в нашей функции
print(function.grad_fn)  # Первым идет умножение на 10
print(function.grad_fn.next_functions[0][0])  # Вторым идет суммирование
print(function.grad_fn.next_functions[0][0].next_functions[0][0])  # Третьи возведение в квадрат
print(function.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])  # Считаем градиент

"""
Чтобы закончить написание градиентного спуска необходимо обновить тензор.
PyTorch не разрешает такую операцию, но мы можем это обойти: мы можем обновить не тензор, 
по которому можно вычислить градиент, а сами данные, которые лежат в этом тензоре.

Методы с нижним подчеркивание на конце в pytorch означают, что результат будет произведен на том объекте, 
к которому применяется метод.
"""
tensor.data -= LR * tensor.grad
print(tensor)
tensor.grad.zero_()
print(tensor)