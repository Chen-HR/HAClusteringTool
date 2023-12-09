# .\Object.py

import numpy

try:
  from . import Calculator
except ImportError:
  import Calculator

class VirtualPoint:
  """The `VirtualPoint` class represents weighted virtual points between two given points.
  ### Attributes (Class Members):
  - `formula` (function): A formula to calculate virtual point weight.
  - `version` (float): The version of the `VirtualPoint` class.
  ### Methods:
  - `__init__`: Initializes a VirtualPoint instance.
  - `point_weight` (property): Returns the point and weight as a tuple.
  - `weightScaling` (property): Returns the scaled point based on weight.
  - `__str__`: Returns a string representation of the virtual point.
  - `__repr__`: Returns a string representation suitable for object reconstruction.
  - `__hash__`: Returns the hash value based on the point and weight.
  ### Parameters (Constructor):
  - `point1` (tuple): Coordinates of the first point.
  - `point2` (tuple): Coordinates of the second point.
  - `limit` (float): Limit for virtual point calculations.
  - `formula` (function): Optional formula for weight calculation.
  ### Exception:
  - `ValueError`: Raised when the weighted_cluster list is empty.
  """
  formula = lambda point1, point2, limit: numpy.power((limit/Calculator.distance(point1, point2)), 1/(limit*numpy.linalg.norm(point1)*numpy.linalg.norm(point2))) # approximately equal to 1
  def __init__(self, point1: tuple, point2: tuple, limit: float, formula=lambda point1, point2, limit: 1):
    """Initializes a VirtualPoint instance.
    ### Parameters (Constructor):
    - `point1` (tuple): Coordinates of the first point.
    - `point2` (tuple): Coordinates of the second point.
    - `limit` (float): Limit for virtual point calculations.
    - `formula` (function): Optional formula for weight calculation.
    """
    self.formula = VirtualPoint.formula if formula is None else formula
    self.limit = limit
    sources = (numpy.array(point1), numpy.array(point2))
    self.sources = (point1, point2)
    self.point = tuple((numpy.array(point1) + numpy.array(point2)) / 2)
    weight = self.formula(point1, point2, limit)
    self.weight = weight if weight > 1 else 0.0
  @property
  def point_weight(self) -> tuple[tuple, float]:
    return (self.point, self.weight)
  @property
  def weightScaling(self) -> tuple:
    return Calculator.scaling(self.point, self.weight)
  def __str__(self) -> str:
    return str(self.point_weight)
  def __repr__(self) -> str:
    return f"VirtualPoint({self.sources[0]}, {self.sources[1]}, {self.limit}, {self.formula})"
  def __hash__(self) -> int:
    return hash(self.point_weight)

if __name__ == '__main__':
  def points_listTuple(points_numpyArray: numpy.array) -> list[tuple]:
    """Converts a NumPy array of points to a list of tuples.
    ### Parameters:
    - `points_numpyArray` (numpy.array): Array of points in NumPy format.
    ### Returns:
    - `list[tuple]`: List of tuples representing points.
    """
    return [tuple(point_numpyArray) for point_numpyArray in points_numpyArray]
  def point_generator(dimension: int, limit_min: int, limit_max: int, size: int) -> numpy.array:
    """Generates random points in a specified range.
    ### Parameters:
    - `dimension` (int): Number of dimensions for the points.
    - `limit_min` (int): Minimum value for each dimension.
    - `limit_max` (int): Maximum value for each dimension.
    - `size` (int): Number of points to generate.
    ### Returns:
    - `numpy.array`: Array of randomly generated points.
    """
    return numpy.random.uniform(limit_min, limit_max, dimension*size).reshape((size, dimension))

  numpy.random.seed(202312)
  dimension = 2
  limit_min  = -10
  limit_max  = -limit_min

  clustering_radius = 5

  pair: list[tuple] = tuple(points_listTuple(point_generator(dimension, limit_min, limit_max, 2)))
  virtualPoint = VirtualPoint(pair[0], pair[1], clustering_radius)

  print(f"virtualPoint: {virtualPoint}")                              # virtualPoint: ((-1.8230532852737227, 7.7722281616661455), 0.0)
  print()                                                             # 
  print(f"virtualPoint.sources: {virtualPoint.sources}")              # virtualPoint.sources: ((1.9212753542717227, 6.343349759950055), (-5.567381924819168, 9.201106563382236))
  print(f"virtualPoint.point: {virtualPoint.point}")                  # virtualPoint.point: (-1.8230532852737227, 7.7722281616661455)
  print(f"virtualPoint.weight: {virtualPoint.weight}")                # virtualPoint.weight: 0.0
  print()                                                             # 
  print(f"virtualPoint.point_weight: {virtualPoint.point_weight}")    # virtualPoint.point_weight: ((-1.8230532852737227, 7.7722281616661455), 0.0)
  print(f"virtualPoint.weightScaling: {virtualPoint.weightScaling}")  # virtualPoint.weightScaling: (-0.0, 0.0)
  print()                                                             # 
  print(f"virtualPoint.str: {virtualPoint.__str__()}")                # virtualPoint.str: ((-1.8230532852737227, 7.7722281616661455), 0.0)
  print(f"virtualPoint.repr: {virtualPoint.__repr__()}")              # virtualPoint.repr: VirtualPoint((1.9212753542717227, 6.343349759950055), (-5.567381924819168, 9.201106563382236), 5, <function VirtualPoint.<lambda> at 0x000002023DAD6020>)
  print(f"virtualPoint.hash: {virtualPoint.__hash__()}")              # virtualPoint.hash: -4648435086066593848
