# .\Calculator.py

import numpy

try:
  from . import Object
except ImportError:
  import Object

def merge_associated_clusters(clusters: list[set]) -> list[set]: 
  """Merges associated clusters within a list of sets.
  ### Parameters:
  - `clusters` (list[set]): List of clusters to merge.
  ### Returns:
  - `list[set]`: List of merged clusters.
  """
  merged = True
  while merged:
    merged = False
    for i in range(len(clusters)):
      for j in range(i+1, len(clusters)):
        if not clusters[i].isdisjoint(clusters[j]):
          clusters[i] |= clusters.pop(j)
          merged = True
          break
      if merged: break
  return clusters

def distance(point1: tuple, point2: tuple) -> float:
  """Scale the coordinates of a point by a given factor.
  ### Parameters:
  - `point` (tuple): Coordinates of the point.
  - `scaling` (int | float): Scaling factor.
  ### Returns:
  - `tuple`: Scaled coordinates of the point.
  """
  return float(numpy.linalg.norm(numpy.array(point1) - numpy.array(point2)))

def scaling(point: tuple, scaling: int | float) -> tuple:
  """Scale the coordinates of a point by a given factor.
  ### Parameters:
  - `point` (tuple): Coordinates of the point.
  - `scaling` (int | float): Scaling factor.
  ### Returns:
  - `tuple`: Scaled coordinates of the point.
  """
  return tuple(v * scaling for v in point)

def center(cluster: list[tuple]) -> tuple:
  """Calculate the center point of a cluster.
  ### Parameters:
  - `cluster` (list[tuple]): List of coordinates representing a cluster.
  ### Returns:
  - `tuple`: Coordinates of the cluster center.
  """
  return tuple(numpy.array([sum(vL) for vL in numpy.transpose(numpy.array(list(cluster)))]) / len(cluster))

def center_of_gravity(weighted_cluster: list[tuple]) -> tuple:
  """Calculate the center of gravity for a set of weighted points.
  ### Parameters:
  - `weighted_cluster` (list[tuple]): List of tuples where each tuple contains a point and its weight.
  ### Returns:
  - `tuple`: Coordinates of the center of gravity.
  ### Exception:
  - `ValueError`: Raised when the weighted_cluster list is empty.
  """
  if not weighted_cluster:
    raise ValueError("The weighted_cluster list is empty.")
  print(f"center_of_gravity({weighted_cluster})")
  cluster: list[tuple] = [(scaling(point[0], point[1])) for point in weighted_cluster if point[1] > 0]
  return center(cluster)

def weight(cluster: tuple[tuple], point: tuple, limit: float | int) -> float:
  """Calculate the weight of a point within a cluster based on virtual point weights.
  ### Parameters:
  - `cluster` (tuple[tuple]): Cluster of points.
  - `point` (tuple): Point for which weight is calculated.
  - `limit` (float | int): Limit for virtual point calculations.
  ### Returns:
  - `float`: Weight of the point within the cluster.
  """
  return sum([Object.VirtualPoint(point, point_i, limit).weight for point_i in cluster if point_i != point]) / len(cluster)

def cluster_centers(clusters: list[tuple[tuple]]) -> list[tuple]:
  """Calculate the center coordinates of multiple clusters.
  ### Parameters:
  - `clusters` (list[tuple[tuple]]): List of clusters, where each cluster is a tuple of points.
  ### Returns:
  - `list[tuple]`: List of coordinates representing the center of each cluster.
  """
  return [center(cluster) for cluster in clusters if cluster]

def cluster_centers_of_gravity(clusters: list[tuple[tuple, float]]) -> list[tuple]:
  """Calculate the center of gravity coordinates for multiple clusters with weights.
  ### Parameters:
  - `clusters` (list[tuple[tuple, float]]): List of clusters, where each cluster is a tuple of points and weights.
  ### Returns:
  - `list[tuple]`: List of coordinates representing the center of gravity of each cluster.
  """
  return [center_of_gravity(cluster) for cluster in clusters if len(cluster) > 0]
