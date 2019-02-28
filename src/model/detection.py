from model import Rectangle


class Detection(Rectangle):
    """
        id: for tracking objects
        label: class
        rectangle: bounding box
    """
    id: str
    label: str

    def __init__(self, id: str, label: str, top_left=(0, 0), width=0, height=0):
        super().__init__(top_left, width, height)
        self.id = id
        self.label = label
