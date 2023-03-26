from sys import argv


x = int(argv[1])
y = int(argv[2])


class NeuronExclusiveOr:

    def __init__(self):
        self.__x = x
        self.__y = y

        """
            Решаем через обратное 'И' + коэффициенты, чтобы передать в пороговую функцию активации корректные данные.
        """
        self.__first_neuron = self.__get_first_neuron_decision()
        self.__second_neuron = self.__get_second_neuron_decision()

    def __get_first_neuron_decision(self):
        if -1 * self.__x + -1 * self.__y + 2 > 0:
            return 1
        else:
            return 0

    def __get_second_neuron_decision(self):
        if -1 * self.__x + -1 * self.__y + 1 > 0:
            return 1
        else:
            return 0

    def get_decision(self):
        if 1 * self.__first_neuron - 1 * self.__second_neuron + 0 > 0:
            return 1
        else:
            return 0


if __name__ == '__main__':
    print(NeuronExclusiveOr().get_decision())
