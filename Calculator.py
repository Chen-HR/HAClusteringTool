import numpy

import Object

def distance(point1: tuple, point2: tuple) -> float:
  """Calculate the distance between two points using `numpy.linalg.norm`

  ### Parameters
  - `point1` (tuple): The coordinates of the first point.
  - `point2` (tuple): The coordinates of the second point.

  ### Returns
  - `float`: The Euclidean distance between the two points.
  """
  return float(numpy.linalg.norm(numpy.array(point1) - numpy.array(point2)))

def center(cluster: set[tuple]) -> tuple[tuple]:
  """Calculate the center point of a cluster

  ### Parameters
  - `cluster` (set[tuple]): A set of points representing the cluster.

  ### Returns
  - `tuple`: The center point coordinates of the cluster.
  """
  return tuple(numpy.array([sum(vL) for vL in numpy.transpose(numpy.array(list(cluster)))]) / len(cluster))

def center_of_gravity(weighted_cluster: dict) -> tuple:
  """Calculate the center of gravity for a set of weighted points.

  ### Parameters
  - `weighted_dict` (dict): A dictionary where keys are points (tuples) and values are weights.

  ### Returns
  - `tuple`: The center of gravity coordinates.
  """
  center_coordinates = [0] * len(next(iter(weighted_cluster.keys())))
  for point, weight in weighted_cluster.items():
    for i, coordinate in enumerate(point):
      center_coordinates[i] += coordinate * (weight / sum(weighted_cluster.values()))
  return tuple(center_coordinates)

def weight(cluster: set[tuple], point: tuple, limit: float | int) -> float:
  """Calculate the weight of a point within a cluster based on virtual point weights.

  This function calculates the weight of a given point within a cluster. The weight is determined based on the virtual
  point weights generated from the distances between the given point and other points in the cluster.

  ### Parameters
  - `cluster` (set[tuple]): The set of points representing a cluster.
  - `point` (tuple): The coordinates of the point for which the weight is calculated.
  - `limit` (float | int): The distance limit used in the virtual point weight calculation.

  ### Returns
  - `float`: The calculated weight of the point within the cluster.
  """
  return sum([Object.VirtualPoint(point, point_i, limit).weight for point_i in cluster.difference((point,))]) / len(cluster)

def cluster_centers(clusters: list[set[tuple]]) -> list[tuple[tuple]]:
 return [center(cluster) for cluster in clusters if cluster]

def cluster_centers_of_gravity(clusters: list[set[tuple]]) -> list[tuple[tuple]]:
 return [center_of_gravity(cluster) for cluster in clusters if cluster]
