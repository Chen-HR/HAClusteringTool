"""
# Integration

## Overview

The "Integration" package provides a collection of functions for clustering and pairing operations based on distance limits and weighted virtual points. The functions support both source points (V1 algorithm) and virtual points (V2 algorithm), offering flexibility in clustering strategies.

## Functions

### `clustering`

```python
def clustering(points: list[tuple], limit: float | int, useVirtualPoints: bool = False, scaling: float | int = 0.8, useSource: bool = True, returnVirtualPoint: bool = False, withWeight: bool = False, weight = None) -> list[set[VirtualPoint | tuple]]:
```

### `pairing`

```python
def pairing(clusters: list[set[tuple]], limit: float | int, useVirtualPoints: bool = False, scaling: float | int = 0.8, useSource: bool = True, withWeight: bool = False, weight = None) -> list[tuple[set[tuple], list[int]]]:
```

### Other Clustering Functions

Several other functions are available for specific clustering scenarios, including variations of the V1 and V2 algorithms, as well as functions for calculating distances, centers, weights, and merging associated clusters.

## Notes

- Ensure that the required dependencies, such as `numpy`, are installed for proper functioning.
- Review the provided example usage for better understanding and application in specific scenarios.

---

Feel free to customize the documentation further based on additional details or specific use cases for your functions.
"""

import numpy

try:
  import Object
  import Calculator
except ModuleNotFoundError:
  from Integration import Object
  from Integration import Calculator
  __all__ = ["Object", "Calculator"]
  version = 1.0

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
  return clustering_V1_universal(points, lambda point1, point2: Calculator.distance(point1, point2) < limit)

def clustering_V2_1(points: list[tuple], limit: float | int, scaling: float | int = 0.8, returnVirtualPoint: bool = False) -> list[set[Object.VirtualPoint | tuple]]:
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
    virtualPoints: list[Object.VirtualPoint] = [Object.VirtualPoint(points[i], points[j], limit) for i in range(len(points)) for j in range(i + 1, len(points))]
    # Generate and return virtual point clusters using existing virtual points as reference targets
    return clustering_V1_universal(virtualPoints, lambda point1, point2: point1.weight > 0 and point2.weight > 0 and Calculator.distance(point1.point, point2.point) < scaling * limit)
  else:
    # Generate weighted virtual points
    virtualPoints: list[tuple] = [virtualPoint.point for virtualPoint in [Object.VirtualPoint(points[i], points[j], limit) for i in range(len(points)) for j in range(i + 1, len(points))] if virtualPoint.weight > 0]
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
  - `weight` (callable, optional): A function for calculating weights. If None, use the default `Calculator.weight` function.

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
  virtualPoints: list[Object.VirtualPoint] = [Object.VirtualPoint(points[i], points[j], limit) for i in range(len(points)) for j in range(i + 1, len(points))]
  # Generate and return virtual point source clusters using existing virtual points as reference targets
  clusters = clustering_V1_universal(virtualPoints, lambda point1, point2: point1.weight > 0 and point2.weight > 0 and Calculator.distance(point1.point, point2.point) < scaling * limit, [virtualPoint.sources for virtualPoint in virtualPoints])
  if withWeight:
    if weight is None:
      weight = Calculator.weight
    clusters = [{point: weight(cluster, point, scaling * limit) for point in cluster} for cluster in clusters]
  return clusters

def clustering_V2(points: list[tuple], limit: float | int, scaling: float | int = 0.8, useSource: bool = True, returnVirtualPoint: bool = False, withWeight: bool = False, weight = None) -> list[set[Object.VirtualPoint | tuple]]:
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
  - `weight` (callable, optional): A function for calculating weights. If None, use the default `Calculator.weight` function.

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

def clustering(points: list[tuple], limit: float | int, useVirtualPoints: bool = False, scaling: float | int = 0.8, useSource: bool = True, returnVirtualPoint: bool = False, withWeight: bool = False, weight = None) -> list[set[Object.VirtualPoint | tuple]]:
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
  - `weight` (callable, optional): A function for calculating weights. If None, use the default `Calculator.weight` function.

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

def pairing(clusters: list[set[tuple]], limit: float | int, withSource: bool = False, useVirtualPoints: bool = False, scaling: float | int = 0.8, useSource: bool = True, withWeight: bool = False, weight = None) -> list[set[tuple]] | list[tuple[set[tuple], list[int]]]:
  """Pair clusters based on a distance limit.

  This method pairs clusters from the input list based on a given distance limit. It uses the `clustering` function to form clusters from the union of points in the input clusters. 
  The pairs are either returned directly or with additional information about the indices of clusters that contributed to each pair.

  ### Parameters
  - `clusters` (list[set[tuple]]): A list of clusters to pair.
  - `limit` (float | int): The distance limit to form pairs. Clusters within this distance will be paired together.
  - `withSource` (bool, optional): If True, return pairs with source cluster indices; if False, return pairs without source cluster indices. Default is False.
  - `useVirtualPoints` (bool, optional): If True, use V2 algorithm with virtual points; if False, use V1 algorithm with source points. Default is False.
  - `scaling` (float | int, optional): Scaling factor for the distance limit when comparing virtual points. Default is 0.8.
  - `useSource` (bool, optional): If True, use source clusters in pairing; if False, use virtual point clusters. Default is True.
  - `withWeight` (bool, optional): If True, include weights in the result; if False, return pairs without weights. Default is False.
  - `weight` (callable, optional): A function for calculating weights. If None, use the default `Calculator.weight` function.

  ### Returns
  - `list[set[tuple]]` or `list[tuple[set[tuple], list[int]]]`: A list of pairs, where each pair is represented as a set of tuples. If 'withSource' is True, each pair is accompanied by a list of indices indicating the contributing clusters.

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
  if withSource == False:
    return pairs
  else:
    # Find the indices of clusters that contributed to each pair
    pairs_from: list[list[int]] = [[j for j in range(len(clusters)) if not pairs[i].isdisjoint(clusters[j])] for i in range(len(pairs))]
    return [(pairs[i], pairs_from[i]) for i in range(len(pairs))]


if __name__ == "__main__":
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
  print("  Cluster Centers (clustering_V1  ):", Calculator.cluster_centers(clustering_V1(point_list, distance_limitation_coefficient)))
  print("  Cluster Centers (clustering     ):", Calculator.cluster_centers(clustering(point_list, distance_limitation_coefficient, useVirtualPoints=False)))
  cluster.append(Calculator.cluster_centers(clustering(point_list, distance_limitation_coefficient, useVirtualPoints=False)))
  print("clustering_V2_1: ")
  print("  Cluster Centers (clustering_V2_1):", Calculator.cluster_centers(clustering_V2_1(point_list, distance_limitation_coefficient, returnVirtualPoint=False)))
  print("  Cluster Centers (clustering_V2  ):", Calculator.cluster_centers(clustering_V2(point_list, distance_limitation_coefficient, useSource=False, returnVirtualPoint=False)))
  print("  Cluster Centers (clustering     ):", Calculator.cluster_centers(clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=False, returnVirtualPoint=False)))
  cluster.append(Calculator.cluster_centers(clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=False, returnVirtualPoint=False)))
  print("clustering_V2_2: ")
  print("  Cluster Centers (clustering_V2_2):", Calculator.cluster_centers(clustering_V2_2(point_list, distance_limitation_coefficient)))
  print("  Cluster Centers (clustering_V2  ):", Calculator.cluster_centers(clustering_V2(point_list, distance_limitation_coefficient, useSource=True)))
  print("  Cluster Centers (clustering     ):", Calculator.cluster_centers(clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=True)))
  cluster.append(Calculator.cluster_centers(clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=True)))
  print("clustering_V2_2: ")
  print("  Cluster Centers (clustering_V2_2):", Calculator.cluster_centers_of_gravity(clustering_V2_2(point_list, distance_limitation_coefficient, withWeight=True)))
  print("  Cluster Centers (clustering_V2  ):", Calculator.cluster_centers_of_gravity(clustering_V2(point_list, distance_limitation_coefficient, useSource=True, withWeight=True)))
  print("  Cluster Centers (clustering     ):", Calculator.cluster_centers_of_gravity(clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=True, withWeight=True)))
  cluster.append(Calculator.cluster_centers_of_gravity(clustering(point_list, distance_limitation_coefficient, useVirtualPoints=True, useSource=True, withWeight=True)))
  print()
  print("cluster:", cluster)
  print("pairing:", pairing(cluster, distance_limitation_coefficient, True))
