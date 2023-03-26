import torch

# Создаем матрицу из нулей размером 3 на 4
zeroes = torch.zeros([3, 4])
print(zeroes)

# Также можно создать произвольную матрицу, передав в нее значения через списки, узнать ее размер
my_tensor = torch.Tensor(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
)
print(my_tensor)
size = my_tensor.size()
print(size)

# По объекту типа Tensor можно проходится с помощью индекса, как по обычному списку
first_elem_in_first_row = my_tensor[0][0]
print(first_elem_in_first_row)

# Чтобы получить первый элемент каждой строки, необходимо сделать следующий срез
first_elem_slice = my_tensor[:, 0]
print(first_elem_slice)

"""
С объектом типа Tensor можно производить арифметические операции.
Операция будет совершена над каждым элементом каждой строки
"""
zeroes_plus_two = zeroes + 2
print(zeroes_plus_two)

"""
Также можно складывать тензоры, но данные тензоры должны иметь одинаковый размер / форму.
Операции производятся поэлементно: 1 элемент 1 строки 1 тензора, например, складывается с 1 элементом 1 строки 2 тензора
"""
comprehension = my_tensor + zeroes_plus_two
print(comprehension)

"""
Помимо базовых операций в torch также имеются функции логарифмов, синусов, косинусов и так далее: 
torch.log(), torch.sin(), torch.cos(), torch.tan(), torch.acrtan()
"""

"""
Также существуют операции сравнения для каждого элемента каждой строки тензора. 
Такой объект называется маской и может быть использован для фильтрации. 
Например, нам нужны только элементы, которые больше 6. На выходе получаем одномерный список.
"""
more_than_6 = my_tensor > 6
print(more_than_6)
my_tensor_filtered_by_mask = my_tensor[more_than_6]
print(my_tensor_filtered_by_mask)

"""
Чтобы скопировать тензор (создать новый объект, а не добавить ссылку на существующий), нужно использовать метод 
torch.clone(). Данный метод работает схожим на copy.deepcopy() метод.
"""
cloned_zeroes = torch.clone(zeroes)
cloned_zeroes[0, 0] = 666
print(cloned_zeroes, zeroes)

"""
Пока что мы работали с дробными числами (внимание на точки в тензорах).
Чтобы определить тип данных в тензоре, нужно обратиться к атрибуту 'dtype' нужного тензора.
Поменять тип данных можно с помощью соответсвующих методов 'double', 'int', 'float'
"""
print(zeroes.dtype)
print(zeroes.int())
print(zeroes.double())

# Torch также поддерживает работу с numpy, чтобы обрабатывать входные данные из numpy
import numpy

numpy_array = numpy.array(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

# Numpy array -> torch Tensor
from_numpy_to_torch = torch.from_numpy(numpy_array)
print(from_numpy_to_torch)

# torch Tensor -> Numpy array
from_torch_to_numpy = from_numpy_to_torch.numpy()
print(from_torch_to_numpy, type(from_torch_to_numpy))

# ------------------------------------------------------------------------------------------------------------------
print('\n\n --------------------------------------------------------------- \n\n')

# Проверяем, есть ли у нас GPU на машине (ПК)
print(f'GPU is available: {torch.cuda.is_available()}')

# Перевод на ту платформу (CPU | GPU), в зависимости от наличия GPU на машине
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
changed_to_device = zeroes.to(device)
