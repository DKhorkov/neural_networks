from sys import argv


x = int(argv[1])
y = int(argv[2])


class NeuronAnd:

    def __init__(self):
        self.__x = x
        self.__y = y

    def get_decision(self):
        if 1 * self.__x + 1 * self.__y - 1 > 0:
            return 1
        else:
            return 0


if __name__ == '__main__':
    print(NeuronAnd().get_decision())
