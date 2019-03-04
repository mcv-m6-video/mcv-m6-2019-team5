from typing import Optional


class Rectangle:
    """
    In the class rectangle we define a rectangle through the top left point, width and height
    
    In the function get_bottom_right with the top-left point and the values for width and height
    we find the bottom_right point
    
    Finally we compute the area of a rectangle in the function get_area 
    
    """

    top_left: (float, float)
    width: float
    height: float

    def __init__(self, top_left=(0, 0), width=0, height=0):
        self.top_left = top_left
        self.width = width
        self.height = height

    def get_bottom_right(self) -> (float, float):
        return self.top_left[0] + self.width, self.top_left[1] + self.height

    def get_bottom_left(self) -> (float, float):
        return self.top_left[0] + self.width, self.top_left[1]

    def get_top_right(self) -> (float, float):
        return self.top_left[0], self.top_left[1] + self.height

    def contains_point(self, point: (float, float)) -> bool:
        return (self.top_left[0] <= point[0] <= self.get_bottom_right()[0] and
                self.top_left[1] <= point[1] <= self.get_bottom_right()[1])

    def get_area(self):
        return self.width * self.height

    def union(self, other: 'Rectangle') -> 'Rectangle':
        rec = Rectangle()
        rec.top_left = (min(self.top_left[0], other.top_left[0]), min(self.top_left[1], other.top_left[1]))
        bottom_right = (max(self.get_bottom_right()[0], other.get_bottom_right()[0]),
                        max(self.get_bottom_right()[1], other.get_bottom_right()[1]))

        rec.width = (bottom_right[0] - rec.top_left[0])
        rec.height = (bottom_right[1] - rec.top_left[1])

        return rec

    def intersection(self, other: 'Rectangle') -> Optional['Rectangle']:
        rec = Rectangle()
        rec.top_left = (max(self.top_left[0], other.top_left[0]), max(self.top_left[1], other.top_left[1]))
        bottom_right = (min(self.get_bottom_right()[0], other.get_bottom_right()[0]),
                        min(self.get_bottom_right()[1], other.get_bottom_right()[1]))

        rec.width = (bottom_right[0] - rec.top_left[0])
        rec.height = (bottom_right[1] - rec.top_left[1])

        return rec

    def iou(self, other: 'Rectangle') -> float:
        return self.intersection(other).get_area() / self.union(other).get_area()

    def __str__(self):
        return 'Rectangle(top_left={0}, width={1}, height={2})'.format(self.top_left, self.width, self.height)


rec1 = Rectangle()
rec2 = Rectangle()

rec1.height = 100
rec1.width = 100
rec1.top_left = 0, 0

rec2.height = 50
rec2.width = 50
rec2.top_left = 25, 25

print(rec1.iou(rec2))