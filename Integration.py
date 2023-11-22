import numpy # numpy-1.26.0

def calculate_distance(point1: tuple, point2: tuple) -> float:
  """Calculate the distance between two points using `numpy.linalg.norm`

  ### Parameters
  - `point1` (tuple): The coordinates of the first point.
  - `point2` (tuple): The coordinates of the second point.

  ### Returns
  - `float`: The Euclidean distance between the two points.
  """
  return float(numpy.linalg.norm(numpy.array(point1) - numpy.array(point2)))

def calculate_center(cluster: set[tuple]) -> tuple[tuple]:
  """Calculate the center point of a cluster

  ### Parameters
  - `cluster` (set[tuple]): A set of points representing the cluster.

  ### Returns
  - `tuple`: The center point coordinates of the cluster.
  """
  return tuple(numpy.array([sum(vL) for vL in numpy.transpose(numpy.array(list(cluster)))]) / len(cluster))

def calculate_center_of_gravity(weighted_cluster: dict) -> tuple:
  """Calculate the center of gravity for a set of weighted points.

  ### Parameters
  - `weighted_dict` (dict): A dictionary where keys are points (tuples) and values are weights.

  ### Returns
  - `tuple`: The center of gravity coordinates.
  """
  total_weight = sum(weighted_cluster.values())
  center_coordinates = [0] * len(next(iter(weighted_cluster.keys())))
  for point, weight in weighted_cluster.items():
    for i, coordinate in enumerate(point):
      center_coordinates[i] += coordinate * (weight / sum(weighted_cluster.values()))
  return tuple(center_coordinates)

def calculate_weight(cluster: set[tuple], point: tuple, limit: float | int) -> float:
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
  return sum([VirtualPoint(point, point_i, limit).weight for point_i in cluster.difference((point,))]) / len(cluster)

def merge_associated_clusters(clusters: list[set]) -> list[set]: 
  """Merges associated clusters within a list of sets.

  ### Parameters
  - `clusters` (list[set]): A list of sets representing clusters.

  ### Returns
  - `list[set]`: A list of sets with associated clusters merged.
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

def clustering_V1_universal(points: list, rule, associated: list[tuple[tuple]] | None = None) -> list[set]:
  """Perform universal clustering on a list of points based on a given clustering rule.

  This algorithm performs universal clustering on a list of points based on a given clustering rule. It allows for
  the specification of an initial set of associated clusters or creates one if not provided. The algorithm iterates
  through all pairs of points and merges clusters if they satisfy the provided clustering rule. Finally, the 
  associated clusters are merged using the 'merge_associated_clusters' method.

  Ensure that the length of 'points' and 'associated' lists is the same. The clustering rule should be a boolean
  function determining whether two points should be clustered together.

  ### Parameters
  - `points` (list): A list of points represented as tuples.
  - `rule` (_type_): A clustering rule that defines whether two points should be grouped together. Should be a boolean function, which receives two points (p1, p2), as `lambda p1, p2: ......`.
  - as`sociated (list[tuple[tuple]] | None, optional): A list of associated clusters. Default is None.

  ### Exception
  - `ValueError`: If the length of 'points' and 'associated' lists are not the same.

  ### Returns
  - `list[set]`: A list of clusters, where each cluster is represented as a set of tuples.

  ### Example Usage
  ```python
  # Example Usage
  points = [(0, 0), (1, 1), (2, 2), (5, 5), (6, 6)]
  rule = lambda p1, p2: abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1
  clusters = V1_universal(points, rule)
  ```
  """
  # 1. Create a list of associated clusters from `points` if `associated` is not provided
  if associated is None: 
    associated = [tuple((point, )) for point in points]

  # 2. Throw an exception `ValueError` if the length of 'points' and 'associated' lists are not the same.
  if len(points) != len(associated): 
    raise ValueError("points and associated must be the same length")

  # 3. Iterate over all point pairs to build clusters based on the provided clustering rules.
  clusters: list[set] = [set(associated[i]) | set(associated[j]) for i in range(len(points)) for j in range(i+1, len(points)) if rule(points[i], points[j])]

  # 4. Merge associated clusters using the 'merge_associated_clusters' method, and return it result.
  return merge_associated_clusters(clusters)

def clustering_V1(points: list[tuple], limit: float | int) -> list[set[tuple]]:
  """Perform clustering on a list of points based on a distance limit.

  This method performs clustering on a list of points based on a given distance limit. It uses the Euclidean distance
  between points to determine whether two points should be grouped together. The clusters are formed by iterating
  through all pairs of points and merging clusters if the distance between them is less than the provided limit.
  Finally, the associated clusters are merged using the 'merge_associated_clusters' method.

  ### Parameters
  - `points` (list[tuple]): A list of points represented as tuples.
  - `limit` (float | int): The distance limit to form clusters. Points within this distance will be grouped together.

  ### Returns
  - `list[set[tuple]]`: A list of clusters, where each cluster is represented as a set of tuples.

  ### Example Usage
  ```python
  # Example Usage
  points = [(0, 0), (1, 1), (2, 2), (5, 5), (6, 6)]
  limit = 1.5
  clusters = V1(points, limit)
  ```
  """
  return clustering_V1_universal(points, lambda point1, point2: calculate_distance(point1, point2) < limit)

class VirtualPoint:
  """
  The `VirtualPoint` class represents weighted virtual points between two given points.

  ### Attributes:
  - `point`: Tuple representing the position of the virtual point (midpoint between two input points).
  - `weight`: Weight of the virtual point calculated using the formula log(1/(𝑑/𝑟)), where 𝑑 is the distance between two points and 𝑟 is the distance limitation coefficient.

  ### Methods:
  - `__init__(self, point1: tuple, point2: tuple, limit: float)`: Initializes a VirtualPoint instance.
    - `point1`: Tuple representing the coordinates of the first point.
    - `point2`: Tuple representing the coordinates of the second point.
    - `limit`: Distance limitation coefficient (𝑟).

  ### Example Usage:
  ```python
  # Example Usage
  point1 = (0, 0)
  point2 = (1, 1)
  limit = 2.0
  virtual_point = VirtualPoint(point1, point2, limit)
  print(virtual_point.point)   # Output: (0.5, 0.5)
  print(virtual_point.weight)  # Output: 0.707107
  ```

  ### Explanation:
  - The `point` attribute is the midpoint between `point1` and `point2`.
  - The `weight` attribute is calculated using the formula log(1/(𝑑/𝑟)), where 𝑑 is the distance between `point1` and `point2`.
  - The `__init__` method initializes the virtual point based on the provided points and distance limitation coefficient.
  """

  def __init__(self, point1: tuple, point2: tuple, limit: float):
    """Initializes a VirtualPoint instance.

    Parameters:
    - `point1`: Tuple representing the coordinates of the first point.
    - `point2`: Tuple representing the coordinates of the second point.
    - `limit`: Distance limitation coefficient (𝑟).
    """
    self.sources = (numpy.array(point1), numpy.array(point2))
    self.point = tuple((self.sources[0] + self.sources[1]) / 2)
    self.weight = numpy.log(1 / (numpy.linalg.norm(self.sources[0] - self.sources[1]) / limit))
    self.sources = (point1, point2)  # Note: This line seems redundant; it assigns the original tuple values again.

def clustering_V2_1(points: list[tuple], limit: float | int, scaling: float | int = 0.8, returnVirtualPoint: bool = False) -> list[set[VirtualPoint | tuple]]:
  """Perform clustering on a list of points based on weighted virtual points with a distance limit.

  This method extends the V1 algorithm by introducing weighted virtual points. Each pair of points generates a virtual
  point with a weight based on the distance between the original points. Clustering is performed on these virtual points
  using the provided clustering rule. The resulting clusters are then merged based on the 'merge_associated_clusters'
  method.

  ### Parameters
  - `points` (list[tuple]): A list of points represented as tuples.
  - `limit` (float | int): The distance limit to form clusters. Points within this distance will be grouped together.
  - `scaling` (float | int, optional): Scaling factor for the distance limit when comparing virtual points. Default is 0.8.
  - `returnVirtualPoint` (bool, optional): If True, return `list[set[VirtualPoint]]`; if False, return `list[set[tuple]]`. Default is False.

  ### Returns
  - `list[set[VirtualPoint | tuple]]`: A list of clusters, where each cluster is represented as a set of VirtualPoint.

  ### Example Usage
  ```python
  # Example Usage
  points = [(0, 0), (1, 1), (2, 2), (5, 5), (6, 6)]
  limit = 1.5
  clusters = V2_1(points, limit)
  ```
  """
  if returnVirtualPoint:
    # Generate weighted virtual points
    virtualPoints: list[VirtualPoint] = [VirtualPoint(points[i], points[j], limit) for i in range(len(points)) for j in range(i + 1, len(points))]
    # Generate and return virtual point clusters using existing virtual points as reference targets
    return clustering_V1_universal(virtualPoints, lambda point1, point2: point1.weight > 0 and point2.weight > 0 and calculate_distance(point1.point, point2.point) < scaling * limit)
  else:
    # Generate weighted virtual points
    virtualPoints: list[tuple] = [virtualPoint.point for virtualPoint in [VirtualPoint(points[i], points[j], limit) for i in range(len(points)) for j in range(i + 1, len(points))] if virtualPoint.weight > 0]
    # Generate and return virtual point clusters using existing virtual points as reference targets
    return clustering_V1(virtualPoints, scaling * limit)

def clustering_V2_2(points: list[tuple], limit: float | int, scaling: float | int = 0.8, withWeight: bool = False, weight = None) -> list[set[tuple] | dict[tuple, float]]:
  """Perform clustering on a list of points based on weighted virtual points with a distance limit, and return source clusters.

  This method extends the V1 algorithm by introducing weighted virtual points. Each pair of points generates a virtual
  point with a weight based on the distance between the original points. Clustering is performed on these virtual points
  using the provided clustering rule. The resulting clusters are then merged based on the 'merge_associated_clusters'
  method. This method returns the source clusters of the virtual points.

  ### Parameters
  - `points` (list[tuple]): A list of points represented as tuples.
  - `limit` (float | int): The distance limit to form clusters. Points within this distance will be grouped together.
  - `scaling` (float | int, optional): Scaling factor for the distance limit when comparing virtual points. Default is 0.8.
  - `withWeight` (bool, optional): If True, include weights in the result; if False, return source clusters without weights. Default is False.
  - `weight` (callable, optional): A function for calculating weights. If None, use the default `calculate_weight` function.

  ### Returns
  - `list[set[tuple]]`: A list of clusters, where each cluster is represented as a set of tuples. Each tuple in a cluster represents the source points of a virtual point.
    If `withWeight` is True, the result is a list of dictionaries, where each dictionary represents a cluster with point weights.

  ### Example Usage
  ```python
  # Example Usage
  points = [(0, 0), (1, 1), (2, 2), (5, 5), (6, 6)]
  limit = 1.5
  source_clusters = V2_2(points, limit)
  ```
  """
  # Generate weighted virtual points
  virtualPoints: list[VirtualPoint] = [VirtualPoint(points[i], points[j], limit) for i in range(len(points)) for j in range(i + 1, len(points))]
  # Generate and return virtual point source clusters using existing virtual points as reference targets
  clusters = clustering_V1_universal(virtualPoints,
                                     lambda point1, point2: point1.weight > 0 and point2.weight > 0 and calculate_distance(point1.point, point2.point) < scaling * limit,
                                     [virtualPoint.sources for virtualPoint in virtualPoints])
  if withWeight:
    if weight is None:
      weight = calculate_weight
    clusters = [{point: weight(cluster, point, scaling * limit) for point in cluster} for cluster in clusters]
  return clusters

def clustering_V2(points: list[tuple], limit: float | int, scaling: float | int = 0.8, useSource: bool = True, returnVirtualPoint: bool = False, withWeight: bool = False, weight = None) -> list[set[VirtualPoint | tuple]]:
  """Perform clustering on a list of points based on weighted virtual points with a distance limit.

  This method extends the V1 algorithm by introducing weighted virtual points. Each pair of points generates a virtual
  point with a weight based on the distance between the original points. Clustering is performed on these virtual points
  using the provided clustering rule. The resulting clusters are then merged based on the 'merge_associated_clusters'
  method. This method returns either the virtual point clusters or the source clusters of the virtual points based on the
  'useSource' parameter.

  ### Parameters
  - `points` (list[tuple]): A list of points represented as tuples.
  - `limit` (float | int): The distance limit to form clusters. Points within this distance will be grouped together.
  - `scaling` (float | int, optional): Scaling factor for the distance limit when comparing virtual points. Default is 0.8.
  - `useSource` (bool, optional): If True, return source clusters (like `V2_2`); if False, return virtual point clusters (like `V2_1`). Default is True.
  - `returnVirtualPoint` (bool, optional): If True, return `list[set[VirtualPoint]]`; if False, return `list[set[tuple]]`. Default is False.
  - `withWeight` (bool, optional): If True, include weights in the result; if False, return clusters without weights. Default is False.
  - `weight` (callable, optional): A function for calculating weights. If None, use the default `calculate_weight` function.

  ### Returns
  - `list[set[VirtualPoint | tuple]]`: A list of clusters, where each cluster is represented as a set of `tuple` or `VirtualPoint`.
    It is the parameter "useSource" that determines whether the elements in each cluster are virtual points or source points.
    The type that determines the data in each virtual point cluster is "returnVirtualPoint".
    If `withWeight` is True, the result is a list of dictionaries, where each dictionary represents a cluster with point weights.

  ### Example Usage
  ```python
  # Example Usage
  points = [(0, 0), (1, 1), (2, 2), (5, 5), (6, 6)]
  limit = 1.5
  clusters = V2(points, limit)
  ```
  """
  if useSource:
    return clustering_V2_2(points, limit, scaling, withWeight, weight)
  else:
    return clustering_V2_1(points, limit, scaling, returnVirtualPoint)

def clustering(points: list[tuple], limit: float | int, useVirtualPoints: bool = False, scaling: float | int = 0.8, useSource: bool = True, returnVirtualPoint: bool = False, withWeight: bool = False, weight = None) -> list[set[VirtualPoint | tuple]]:
  """
  Perform clustering on a list of points based on either virtual points or source points with a distance limit.

  This method allows clustering based on either the V1 algorithm (using source points) or the V2 algorithm
  (introducing weighted virtual points). The choice is determined by the 'useVirtualPoints' parameter.

  ### Parameters
  - `points` (list[tuple]): A list of points represented as tuples.
  - `limit` (float | int): The distance limit to form clusters. Points within this distance will be grouped together.
  - `useVirtualPoints` (bool, optional): If True, use V2 algorithm with virtual points; if False, use V1 algorithm with source points. Default is False.
  - `scaling` (float | int, optional): Scaling factor for the distance limit when comparing virtual points. Default is 0.8.
  - `useSource` (bool, optional): If True, return source clusters; if False, return virtual point clusters. Default is True.
  - `returnVirtualPoint` (bool, optional): If True, return `list[set[VirtualPoint]]` in the result; if False, return `list[set[tuple]]`. Applicable when using virtual points. Default is False.
  - `withWeight` (bool, optional): If True, include weights in the result; if False, return clusters without weights. Default is False.
  - `weight` (callable, optional): A function for calculating weights. If None, use the default `calculate_weight` function.

  ### Returns
  - `list[set[VirtualPoint | tuple]]`: A list of clusters, where each cluster is represented as a set of `tuple` or `VirtualPoint`.
    The type that determines the data in each virtual point cluster is 'useSource' and 'returnVirtualPoint' parameters.
    If 'withWeight' is True, the result is a list of dictionaries, where each dictionary represents a cluster with point weights.

  ### Example Usage
  ```python
  # Example Usage
  points = [(0, 0), (1, 1), (2, 2), (5, 5), (6, 6)]
  limit = 1.5
  clusters = clustering(points, limit, useVirtualPoints=True, useSource=False, returnVirtualPoint=True, withWeight=True)
  ```
  """
  if useVirtualPoints:
    return clustering_V2(points, limit, scaling, useSource, returnVirtualPoint, withWeight, weight)
  else:
    return clustering_V1(points, limit)
def pairing(clusters: list[set[tuple]], limit: float | int, useVirtualPoints: bool = False, scaling: float | int = 0.8, useSource: bool = True, withWeight: bool = False, weight = None) -> list[tuple[set[tuple], list[int]]]:
  """Pair clusters based on a distance limit, considering either source points or virtual points.

  This function pairs clusters based on the provided distance limit. It uses the `clustering` function to perform
  clustering either with source points (V1) or virtual points (V2), depending on the `useVirtualPoints` parameter.
  The resulting pairs are represented as tuples, where the first element is the paired cluster and the second
  element is a list of indices indicating which original clusters contributed to the pair.

  ### Parameters
  - `clusters` (list[set[tuple]]): A list of clusters to pair.
  - `limit` (float | int): The distance limit to form pairs. Clusters within this distance will be paired together.
  - `useVirtualPoints` (bool, optional): If True, use V2 algorithm with virtual points; if False, use V1 algorithm with source points. Default is False.
  - `scaling` (float | int, optional): Scaling factor for the distance limit when comparing virtual points. Default is 0.8.
  - `useSource` (bool, optional): If True, use source clusters in pairing; if False, use virtual point clusters. Default is True.
  - `withWeight` (bool, optional): If True, include weights in the result; if False, return pairs without weights. Default is False.
  - `weight` (callable, optional): A function for calculating weights. If None, use the default `calculate_weight` function.

  ### Returns
  - `list[tuple[set[tuple], list[int]]]`: A list of pairs, where each pair is represented as a tuple containing a paired cluster and a list of indices indicating which original clusters contributed to the pair.

  ### Example Usage
  ```python
  # Example Usage
  clusters = [{(0, 0), (1, 1)}, {(2, 2), (5, 5)}, {(6, 6)}]
  limit = 2.0
  paired_clusters = pairing(clusters, limit, useVirtualPoints=True, useSource=False, withWeight=True)
  ```
  """
  # Merge clusters into pairs
  pairs: list[set[tuple]] = clustering(list(set().union(*clusters)), limit, useVirtualPoints=useVirtualPoints, scaling=scaling, useSource=useSource, withWeight=withWeight, weight=weight)
  # Find the indices of clusters that contributed to each pair
  pairs_from: list[list[int]] = [[j for j in range(len(clusters)) if not pairs[i].isdisjoint(clusters[j])] for i in range(len(pairs))]
  return [(pairs[i], pairs_from[i]) for i in range(len(pairs))]


def main():
  # Example usage
  dispersion = 10
  size = 1000
  concentrated = 10
  concentrated_distance = 2

  point_list = [tuple(point) for point in list(numpy.transpose([numpy.random.uniform(-size, size, dispersion), numpy.random.uniform(-size, size, dispersion), numpy.random.uniform(-size, size, dispersion)])) + list(numpy.random.uniform(-size, size, 3) + numpy.transpose([numpy.random.uniform(-concentrated_distance, concentrated_distance, concentrated), numpy.random.uniform(-concentrated_distance, concentrated_distance, concentrated), numpy.random.uniform(-concentrated_distance, concentrated_distance, concentrated)]))]
  distance_limitation_coefficient = concentrated_distance
  cluster: list[list[tuple]] = []

  print("point_list:", point_list)
  print()
  print("clustering_V1: ")
  print("  Cluster Centers (clustering_V1  ):", [calculate_center(cluster) for cluster in clustering_V1(point_list, distance_limitation_coefficient) if cluster])
  print("  Cluster Centers (clustering     ):", [calculate_center(cluster) for cluster in clustering(point_list, distance_limitation_coefficient, useVirtualPoints=False) if cluster])
  cluster.append([calculate_center(cluster) for cluster in clustering(point_list, distance_limitation_coefficient, useVirtualPoints=False) if cluster])
  print("clustering_V2_1: ")
  print("  Cluster Centers (clustering_V2_1):", [calculate_center(cluster) for cluster in clustering_V2_1(point_list, distance_limitation_coefficient, returnVirtualPoint=False) if cluster])
  print("  Cluster Centers (clustering_V2  ):", [calculate_center(cluster) for cluster in clustering_V2(point_list, distance_limitation_coefficient, useSource=False, returnVirtualPoint=False) if cluster])
  print("  Cluster Centers (clustering     ):", [calculate_center(cluster) for cluster in clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=False, returnVirtualPoint=False) if cluster])
  cluster.append([calculate_center(cluster) for cluster in clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=False, returnVirtualPoint=False) if cluster])
  print("clustering_V2_2: ")
  print("  Cluster Centers (clustering_V2_2):", [calculate_center(cluster) for cluster in clustering_V2_2(point_list, distance_limitation_coefficient) if cluster])
  print("  Cluster Centers (clustering_V2  ):", [calculate_center(cluster) for cluster in clustering_V2(point_list, distance_limitation_coefficient, useSource=True) if cluster])
  print("  Cluster Centers (clustering     ):", [calculate_center(cluster) for cluster in clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=True) if cluster])
  cluster.append([calculate_center(cluster) for cluster in clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=True) if cluster])
  print("clustering_V2_2: ")
  print("  Cluster Centers (clustering_V2_2):", [calculate_center_of_gravity(cluster) for cluster in clustering_V2_2(point_list, distance_limitation_coefficient, withWeight=True) if cluster])
  print("  Cluster Centers (clustering_V2  ):", [calculate_center_of_gravity(cluster) for cluster in clustering_V2(point_list, distance_limitation_coefficient, useSource=True, withWeight=True) if cluster])
  print("  Cluster Centers (clustering     ):", [calculate_center_of_gravity(cluster) for cluster in clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=True, withWeight=True) if cluster])
  cluster.append([calculate_center_of_gravity(cluster) for cluster in clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=True, withWeight=True) if cluster])
  print()
  print("cluster:", cluster)
  print("pairing:", pairing(cluster, distance_limitation_coefficient))

if __name__ == "__main__":
  main()
