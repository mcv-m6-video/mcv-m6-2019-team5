class IDGeneratorClass:

    def __init__(self):
        self.value = 1

    def next(self) -> int:
        ret = self.value
        self.value += 1
        return ret


instance = IDGeneratorClass()
