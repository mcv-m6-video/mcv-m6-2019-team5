from model import Rectangle

class Detection(Rectangle):
    """
        id: int. For tracking objects
        label: str. Class of the object
    """

    def __init__(self, id: str, label: str, confidence: float = None, top_left=(0, 0), width=0, height=0):
        super().__init__(top_left, width, height)
        self.id = id
        self.label = label
        self.confidence = confidence

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Detection(id={0}, label={1}, confidence=(2), rectangle={3})'.format(self.id, self.label, self.confidence, super().__str__())
