import torch


LR = 0.001
tensor = torch.tensor(
    [
        [8, 8]
    ],
    requires_grad=True,
    dtype=torch.float
)

tensor_for_optimizer = torch.tensor(
    [
        [8, 8]
    ],
    requires_grad=True,
    dtype=torch.float
)

"""
Ручной спуск, написанный самостоятельно
"""


def parabola_function(variable):
    return 10 * (variable ** 2).sum()


def do_gradient_step(function_for_step, variable):
    """
    Данная функция совершает градиентный шаг и обновляет переменную (как правило тензор), для которой был сделан шаг.

    :param function_for_step: Функция градиентного шага для расчета потерь
    :param variable: Переменная (как правило тензор) для функции
    :return: None
    """
    function_result = function_for_step(variable)
    function_result.backward()
    variable.data -= LR * variable.grad
    variable.grad.zero_()


for _ in range(500):
    do_gradient_step(parabola_function, tensor)

print(tensor)


"""
Спуск через оптимайзер.
Объект, который знает, как лучше делать шаг градиентного спуска. В данном случая стохатический градиентный спуск.

Оптимайзер является оберткой для тензора и совершает с помощью своих методов операции над ним.
"""
optimizer = torch.optim.SGD(params=[tensor], lr=LR)


def do_optimizer_gradient_step(function_for_step, variable):
    """
    Данная функция совершает градиентный шаг и обновляет переменную (как правило тензор), для которой был сделан шаг.

    :param function_for_step: Функция градиентного шага для расчета потерь
    :param variable: Переменная (как правило тензор) для функции
    :return: None
    """
    function_result = function_for_step(variable)
    function_result.backward()
    optimizer.step()  # Совершение градиентного шага после взятия производной
    optimizer.zero_grad()  # Обновление данных тензора после шага


for _ in range(500):
    do_gradient_step(parabola_function, tensor_for_optimizer)

print(tensor_for_optimizer)
