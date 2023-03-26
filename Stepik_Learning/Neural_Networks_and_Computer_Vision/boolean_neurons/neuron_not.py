from sys import argv


x = int(argv[1])


class NeuronNot:

    def __init__(self):
        self.__x = x

    def get_decision(self):
        if -1 * self.__x + 1 > 0:
            return 1
        else:
            return 0


if __name__ == '__main__':
    print(NeuronNot().get_decision())
