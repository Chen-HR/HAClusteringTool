"""
# HAClusteringTool
## Overview

"""
# %%
import numpy # Version: 1.26.0

try:
  from . import Object
  from . import Calculator
except ImportError:
  import Object
  import Calculator

__all__ = ["Object", "Calculator"]
version = 1.1

if __name__ == '__main__':
  import matplotlib.pyplot # Version: 3.8.1

# %%
def clustering_template(points: list, rule, associated: list[tuple[tuple]] | None = None) -> list[tuple]:
  """Generic template for "Hierarchical Aggregation Clustering" using "Single Linkage Agglomerative Algorithm"
  ### Parameters
  - `points` (list): A list of points represented as tuples.
  - `rule` (callable): A clustering rule that defines whether two points should be grouped together.
                       Should be a boolean function, which receives two points (p1, p2), as `lambda p1, p2: ......`.
  - `associated` (list[tuple[tuple]] | None, optional): A list of associated clusters. Default is None.
  ### Exception
  - `ValueError`: If the length of 'points' and 'associated' lists are not the same.
  ### Returns
  - `list[set]`: A list of clusters, where each cluster is represented as a set of tuples.
  """
  # 1. Create a list of associated clusters from `points` if `associated` is not provided
  if associated is None: 
    associated = [tuple((point, )) for point in points]
  # 2. Throw an exception `ValueError` if the length of 'points' and 'associated' lists are not the same.
  if len(points) != len(associated): 
    raise ValueError("points and associated must be the same length")
  # 3. Iterate over all point pairs to build clusters based on the provided clustering rules.
  clusters: list[set] = [set(associated[i]) | set(associated[j]) 
                          for i in range(len(points)) 
                            for j in range(i+1, len(points)) 
                              if rule(points[i], points[j])]
  # 4. Merge associated clusters using the 'Calculator.merge_associated_clusters' method, and return its result.
  return [tuple(s) for s in Calculator.merge_associated_clusters(clusters)]

def clustering_useSource(points: list[tuple], limit: float | int) -> list[tuple]:
  """Perform clustering based on source points within a specified limit.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  ### Returns
  - `list[set[tuple]]`: List of clusters where each cluster is represented as a set of tuples.
  """
  return clustering_template(points, 
                             lambda point1, point2: Calculator.distance(point1, point2) < limit)
def clustering_useSource_centers(points: list[tuple], limit: float | int) -> list[set[tuple]]:
  """Calculate the center coordinates of clusters based on source points within a specified limit.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  ### Returns
  - `list[set[tuple]]`: List of cluster centers.
  """
  return Calculator.cluster_centers(clustering_useSource(points, limit))

def clustering_useVirtual_associateSource(points: list[tuple], limit: float | int, withWeight: bool = False, virtualWeight = None, sourceWeight = None) -> list[tuple[tuple]]:
  """Perform clustering using virtual points associated with source points within a specified limit.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  - `withWeight` (bool, optional): Whether to consider weights. Default is False.
  - `virtualWeight`: Custom formula for virtual point weighting. Default is None.
  - `sourceWeight`: Custom formula for source point weighting. Default is None.
  ### Returns
  - `list[tuple[tuple]]`: List of clustered points.
  """
  virtualWeight = Object.VirtualPoint.formula if virtualWeight is None else virtualWeight
  # Generate weighted virtual points
  virtualPoints_listVirtualPoint: list[Object.VirtualPoint] = [Object.VirtualPoint(points[i], points[j], limit, virtualWeight) 
                                                for i in range(len(points)) 
                                                  for j in range(i + 1, len(points)) 
                                                    # if Calculator.distance(points[i], points[j]) > limit
                                                    ]
  
  virtualPoints = [virtualPoint for virtualPoint in virtualPoints_listVirtualPoint if virtualPoint.weight > 0]
  # Generate virtual point source clusters using existing virtual points as reference targets
  clusters: list[tuple] = clustering_template(virtualPoints, lambda point1, point2: point1.weight > 0 and point2.weight > 0 and Calculator.distance(point1.point, point2.point) < limit, [virtualPoint.sources for virtualPoint in virtualPoints])
  # Adjust output format
  if withWeight:
    sourceWeight = Calculator.weight if sourceWeight is None else sourceWeight
    clusters = [tuple((point, sourceWeight(cluster, point, limit)) for point in cluster) for cluster in clusters]
  return clusters
def clustering_useVirtual_associateSource_centers(points: list[tuple], limit: float | int, withWeight: bool = False, virtualWeight = None, sourceWeight = None) -> list[tuple]:
  """Calculate the center coordinates of clusters based on virtual points associated with source points within a specified limit.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  - `withWeight` (bool, optional): Whether to consider weights. Default is False.
  - `virtualWeight`: Custom formula for virtual point weighting. Default is None.
  - `sourceWeight`: Custom formula for source point weighting. Default is None.
  ### Returns
  - `list[tuple]`: List of cluster centers.
  """
  if withWeight:
    return Calculator.cluster_centers_of_gravity(clustering_useVirtual_associateSource(points, limit, withWeight=withWeight, virtualWeight=virtualWeight, sourceWeight=sourceWeight))
  else:
    return Calculator.cluster_centers(clustering_useVirtual_associateSource(points, limit, virtualWeight=virtualWeight, sourceWeight=sourceWeight))

def clustering_useVirtual_associateVirtual(points: list[tuple], limit: float | int, withWeight: bool = False, virtualWeight = None) -> list[tuple[tuple]]:
  """Perform clustering using virtual points associated with virtual points within a specified limit.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  - `withWeight` (bool, optional): Whether to consider weights. Default is False.
  - `virtualWeight`: Custom formula for virtual point weighting. Default is None.
  ### Returns
  - `list[tuple[tuple]]`: List of clustered points.
  """
  virtualWeight = Object.VirtualPoint.formula if virtualWeight is None else virtualWeight
  # Generate weighted virtual points
  virtualPoints_listVirtualPoint: list[Object.VirtualPoint] = [Object.VirtualPoint(points[i], points[j], limit, virtualWeight) 
                                                for i in range(len(points)) 
                                                  for j in range(i + 1, len(points)) 
                                                    # if Calculator.distance(points[i], points[j]) > limit
                                                    ]
  virtualPoints = [virtualPoint for virtualPoint in virtualPoints_listVirtualPoint if virtualPoint.weight>0]
  # Generate virtual point clusters using existing virtual points as reference targets
  clusters: list[tuple[Object.VirtualPoint]] = clustering_template(virtualPoints, lambda point1, point2: point1.weight > 0 and point2.weight > 0 and Calculator.distance(point1.point, point2.point) < limit)
  # Adjust output format
  if withWeight:
    return [tuple((point.point, point.weight) for point in cluster) for cluster in clusters]
  else:
    return [tuple(point.point for point in cluster) for cluster in clusters]
def clustering_useVirtual_associateVirtual_centers(points: list[tuple], limit: float | int, withWeight: bool = False, virtualWeight = None, sourceWeight = None) -> list[tuple]:
  """Calculate the center coordinates of clusters based on virtual points associated with virtual points within a specified limit.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  - `withWeight` (bool, optional): Whether to consider weights. Default is False.
  - `virtualWeight`: Custom formula for virtual point weighting. Default is None.
  - `sourceWeight`: Custom formula for source point weighting. Default is None.
  ### Returns
  - `list[tuple]`: List of cluster centers.
  """
  if withWeight:
    return Calculator.cluster_centers_of_gravity(clustering_useVirtual_associateVirtual(points, limit, withWeight=withWeight, virtualWeight=virtualWeight))
  else:
    return Calculator.cluster_centers(clustering_useVirtual_associateVirtual(points, limit, virtualWeight=virtualWeight))

def clustering_useVirtual(points: list[tuple], limit: float | int, associateVirtual: bool = False , withWeight: bool = False, virtualWeight = None, sourceWeight = None) -> list[tuple[tuple]]:
  """Perform clustering using virtual points within a specified limit.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  - `associateVirtual` (bool, optional): Whether to associate virtual points. Default is False.
  - `withWeight` (bool, optional): Whether to consider weights. Default is False.
  - `virtualWeight`: Custom formula for virtual point weighting. Default is None.
  - `sourceWeight`: Custom formula for source point weighting. Default is None.
  ### Returns
  - `list[tuple[tuple]]`: List of clustered points.
  """
  if associateVirtual:
    return clustering_useVirtual_associateVirtual(points, limit, withWeight=withWeight, virtualWeight=virtualWeight)
  else:
    return clustering_useVirtual_associateSource(points, limit, withWeight=withWeight, virtualWeight=virtualWeight, sourceWeight=sourceWeight)
def clustering_useVirtual_centers(points: list[tuple], limit: float | int, associateVirtual: bool = False , withWeight: bool = False, virtualWeight = None, sourceWeight = None) -> list[tuple]:
  """Calculate the center coordinates of clusters based on virtual points within a specified limit.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  - `associateVirtual` (bool, optional): Whether to associate virtual points. Default is False.
  - `withWeight` (bool, optional): Whether to consider weights. Default is False.
  - `virtualWeight`: Custom formula for virtual point weighting. Default is None.
  - `sourceWeight`: Custom formula for source point weighting. Default is None.
  ### Returns
  - `list[tuple]`: List of cluster centers.
  """
  if associateVirtual:
    return clustering_useVirtual_associateVirtual_centers(points, limit, withWeight=withWeight, virtualWeight=virtualWeight)
  else:
    return clustering_useVirtual_associateSource_centers(points, limit, withWeight=withWeight, virtualWeight=virtualWeight, sourceWeight=sourceWeight)

def clustering(points: list[tuple], limit: float | int, useVirtual: bool = False, associateVirtual: bool = False , withWeight: bool = False, virtualWeight = None, sourceWeight = None) -> list[tuple[tuple]]:
  """Perform clustering based on specified parameters.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  - `useVirtual` (bool, optional): Whether to use virtual points. Default is False.
  - `associateVirtual` (bool, optional): Whether to associate virtual points. Default is False.
  - `withWeight` (bool, optional): Whether to consider weights. Default is False.
  - `virtualWeight`: Custom formula for virtual point weighting. Default is None.
  - `sourceWeight`: Custom formula for source point weighting. Default is None.
  ### Returns
  - `list[tuple[tuple]]`: List of clustered points.
  """
  if useVirtual:
    return clustering_useVirtual(points, limit, associateVirtual=associateVirtual, withWeight=withWeight, virtualWeight=virtualWeight, sourceWeight=sourceWeight)
  else:
    return clustering_useSource(points, limit)
def clustering_centers(points: list[tuple], limit: float | int, useVirtual: bool = False, associateVirtual: bool = False , withWeight: bool = False, virtualWeight = None, sourceWeight = None) -> list[tuple]:
  """Calculate the center coordinates of clusters based on specified parameters.
  ### Parameters
  - `points` (list[tuple]): List of source points.
  - `limit` (float | int): Limit for clustering.
  - `useVirtual` (bool, optional): Whether to use virtual points. Default is False.
  - `associateVirtual` (bool, optional): Whether to associate virtual points. Default is False.
  - `withWeight` (bool, optional): Whether to consider weights. Default is False.
  - `virtualWeight`: Custom formula for virtual point weighting. Default is None.
  - `sourceWeight`: Custom formula for source point weighting. Default is None.
  ### Returns
  - `list[tuple]`: List of cluster centers.
  """
  if useVirtual:
    return clustering_useVirtual_centers(points, limit, associateVirtual=associateVirtual, withWeight=withWeight, virtualWeight=virtualWeight, sourceWeight=sourceWeight)
  else:
    return clustering_useSource_centers(points, limit)

def pairing(clusters: list[set[tuple]], limit: float | int, withSource: bool = False, useVirtual: bool = False, associateVirtual: bool = False , withWeight: bool = False, virtualWeight = None, sourceWeight = None):
  """Perform pairing of clusters and return the results.
  ### Parameters
  - `clusters` (list[set[tuple]]): List of clusters.
  - `limit` (float | int): Limit for clustering.
  - `withSource` (bool, optional): Whether to include source points in the result. Default is False.
  - `useVirtual` (bool, optional): Whether to use virtual points. Default is False.
  - `associateVirtual` (bool, optional): Whether to associate virtual points. Default is False.
  - `withWeight` (bool, optional): Whether to consider weights. Default is False.
  - `virtualWeight`: Custom formula for virtual point weighting. Default is None.
  - `sourceWeight`: Custom formula for source point weighting. Default is None.
  ### Returns
  - `list[tuple]`: List of paired clusters.
  """
  # Merge clusters into groups
  groups: list[tuple[tuple]] = clustering(list(set().union(*clusters)), limit, useVirtual, associateVirtual, withWeight, virtualWeight, sourceWeight)
  if withSource:
    # Find the indices of clusters that contributed to each pair
    group_from: list[tuple[int]] = [(j for j in range(len(clusters)) if groups[i] in clusters[j]) for i in range(len(groups))]
    return [(groups[i], group_from[i]) for i in range(len(groups))]
  return groups

# %%
if __name__ == "__main__":
  import time
  class Timer:
    """Simple timer class to measure elapsed time."""
    def __init__(self):
      self.starttime = 0.0
    def start(self):
      """Start the timer."""
      self.starttime = time.time()
    def now(self):
      """Get the elapsed time since starting the timer."""
      return time.time() - self.starttime
  timer = Timer()

  def points_listTuple(points_numpyArray: numpy.array) -> list[tuple]:
    """Convert a NumPy array of points to a list of tuples.
    ### Parameters
    - `points_numpyArray` (numpy.array): NumPy array representing points.
    ### Returns
    - `list[tuple]`: List of tuples representing points.
    """
    return [tuple(point_numpyArray) for point_numpyArray in points_numpyArray]
  def noise_generator(dimension: int, min: int, max: int, size: int) -> numpy.array:
    """Generate a NumPy array of random noise points.
    ### Parameters
    - `dimension` (int): Dimensionality of the points.
    - `min` (int): Minimum value for each coordinate.
    - `max` (int): Maximum value for each coordinate.
    - `size` (int): Number of noise points to generate.
    ### Returns
    - `numpy.array`: Array of random noise points.
    """
    return numpy.random.uniform(min, max, dimension*size).reshape((size, dimension))
  def core_generator(dimension: int, min: int, max: int, size: int, distance: int) -> numpy.array:
    """Generate a NumPy array of core points with random noise.
    ### Parameters
    - `dimension` (int): Dimensionality of the points.
    - `min` (int): Minimum value for each coordinate.
    - `max` (int): Maximum value for each coordinate.
    - `size` (int): Number of core points to generate.
    - `distance` (int): Maximum distance for random noise.
    ### Returns
    - `numpy.array`: Array of core points with random noise.
    """
    core = numpy.random.uniform(min, max, dimension)
    return numpy.array([core + diff for diff in noise_generator(dimension, -distance, distance, size)]) 
  def testingData_generator(dimension: int, noise_size: int, cores_number: int, core_size: int, core_distance: int, min: int, max: int) -> numpy.array:
    """Generate testing data with noise and core points.
    ### Parameters
    - `dimension` (int): Dimensionality of the points.
    - `noise_size` (int): Number of noise points.
    - `cores_number` (int): Number of core groups.
    - `core_size` (int): Number of points in each core group.
    - `core_distance` (int): Maximum distance for random noise in core groups.
    - `min` (int): Minimum value for each coordinate.
    - `max` (int): Maximum value for each coordinate.
    ### Returns
    - `numpy.array`: Array of testing data points.
    """
    result: numpy.array = noise_generator(dimension, min, max, noise_size)
    for _ in range(cores_number): result = numpy.concatenate((result, core_generator(dimension, min, max, core_size, core_distance)), axis=0)
    return result
  def testingData_listTuple_generator(dimension: int, noise_size: int, cores_number: int, core_size: int, core_distance: int, min: int, max: int) -> list[tuple]:
    """Generate testing data with noise and core points as a list of tuples.
    ### Parameters
    - `dimension` (int): Dimensionality of the points.
    - `noise_size` (int): Number of noise points.
    - `cores_number` (int): Number of core groups.
    - `core_size` (int): Number of points in each core group.
    - `core_distance` (int): Maximum distance for random noise in core groups.
    - `min` (int): Minimum value for each coordinate.
    - `max` (int): Maximum value for each coordinate.
    ### Returns
    - `list[tuple]`: List of testing data points represented as tuples.
    """
    return points_listTuple(testingData_generator(dimension, noise_size, cores_number, core_size, core_distance, min, max))
  
  def Demo_All_V1(points_list: list[tuple], clustering_radius: int | float):
    """Demonstrate the usage of clustering functions and display results.
    ### Parameters
    - `points_list` (list[tuple]): List of points.
    - `clustering_radius` (int | float): Radius for clustering.
    """
    cluster: list[list[tuple]] = []
    print("points_list:", points_list)
    print()
    
    print("clustering_useSource: ")
    timer.start()
    # print("  Cluster:", clustering_useSource(points_list, clustering_radius))
    print("  Cluster Centers (clustering_V1  ):", clustering_useSource_centers(points_list, clustering_radius))
    print("  Cluster Centers (clustering     ):", clustering_centers(points_list, clustering_radius))
    print(f"  Clustering used {timer.now()} secend")
    cluster.append(clustering_useSource_centers(points_list, clustering_radius))
    
    print("clustering_useVirtual_associateSource: ")
    timer.start()
    # print("  Cluster:", clustering_useVirtual_associateSource(points_list, clustering_radius))
    print("  Cluster Centers (clustering_V2_1):", clustering_useVirtual_associateSource_centers(points_list, clustering_radius))
    print("  Cluster Centers (clustering_V2  ):", clustering_useVirtual_centers(points_list, clustering_radius))
    print("  Cluster Centers (clustering     ):", clustering_centers(points_list, clustering_radius, useVirtual=True))
    print(f"  Clustering used {timer.now()} secend")
    cluster.append(clustering_useVirtual_associateSource_centers(points_list, clustering_radius))
    print("clustering_useVirtual_associateSource_withWeight: ")
    timer.start()
    # print("  Cluster:", clustering_useVirtual_associateSource(points_list, clustering_radius, withWeight=True))
    print("  Cluster Centers (clustering_V2_1):", clustering_useVirtual_associateSource_centers(points_list, clustering_radius, withWeight=True))
    print("  Cluster Centers (clustering_V2  ):", clustering_useVirtual_centers(points_list, clustering_radius, withWeight=True))
    print("  Cluster Centers (clustering     ):", clustering_centers(points_list, clustering_radius, useVirtual=True, withWeight=True))
    print(f"  Clustering used {timer.now()} secend")
    cluster.append(clustering_useVirtual_associateSource_centers(points_list, clustering_radius, withWeight=True))

    print("clustering_useVirtual_associateVirtual: ")
    timer.start()
    # print("  Cluster:", clustering_useVirtual_associateVirtual(points_list, clustering_radius))
    print("  Cluster Centers (clustering_V2_1):", clustering_useVirtual_associateVirtual_centers(points_list, clustering_radius))
    print("  Cluster Centers (clustering_V2  ):", clustering_useVirtual_centers(points_list, clustering_radius, associateVirtual=True))
    print("  Cluster Centers (clustering     ):", clustering_centers(points_list, clustering_radius, useVirtual=True, associateVirtual=True))
    print(f"  Clustering used {timer.now()} secend")
    cluster.append(clustering_useVirtual_associateVirtual_centers(points_list, clustering_radius))
    print("clustering_useVirtual_associateVirtual_withWeight: ")
    timer.start()
    # print("  Cluster:", clustering_useVirtual_associateVirtual(points_list, clustering_radius, withWeight=True))
    print("  Cluster Centers (clustering_V2_1):", clustering_useVirtual_associateVirtual_centers(points_list, clustering_radius, withWeight=True))
    print("  Cluster Centers (clustering_V2  ):", clustering_useVirtual_centers(points_list, clustering_radius, associateVirtual=True, withWeight=True))
    print("  Cluster Centers (clustering     ):", clustering_centers(points_list, clustering_radius, useVirtual=True, associateVirtual=True, withWeight=True))
    print(f"  Clustering used {timer.now()} secend")
    cluster.append(clustering_useVirtual_associateVirtual_centers(points_list, clustering_radius, withWeight=True))
    
    print("cluster:", cluster)
    timer.start()
    print("pairing:", pairing(cluster, clustering_radius))
    print(f"  Clustering used {timer.now()} secend")
  def Demo_MatplotlibPlot_V1(points_list: list[tuple], clustering_radius: int | float, size_min: int | float, size_max: int | float):
    """Demonstrate the usage of clustering functions and visualize results using Matplotlib.
    ### Parameters
    - `points_list` (list[tuple]): List of points.
    - `clustering_radius` (int | float): Radius for clustering.
    - `size_min` (int | float): Minimum size for plotting.
    - `size_max` (int | float): Maximum size for plotting.
    """
    source_label = "source point"
    source_color = "b"
    source_alpha = 0.2
    source_radius = clustering_radius/2
    source_radius_color = source_color
    source_radius_alpha = 0.1

    virtual_label = "virtual point"
    virtual_color = "y"
    virtual_alpha = 0.3
    virtual_radius = clustering_radius/2
    virtual_radius_color = virtual_color
    virtual_radius_alpha = 0.075

    weight_label = "weight point"
    weight_color = "y"
    weight_alpha = 0.3
    weight_radius = clustering_radius/2
    weight_radius_color = weight_color
    weight_radius_alpha = 0.05

    figure: matplotlib.pyplot.Figure = matplotlib.pyplot.figure()
    axes: matplotlib.pyplot.Axes = figure.add_subplot(111)
    axes.set_title("Clustering simulation")
    axes.set_xlim(size_min, size_max)
    axes.set_ylim(size_min, size_max)

    # source point
    axes.scatter(*list(zip(*points_list)), c=source_color, alpha=source_alpha, label=source_label)
    [axes.add_patch(matplotlib.pyplot.Circle(point, source_radius, edgecolor=source_radius_color, facecolor=source_radius_color, alpha=source_radius_alpha)) for point in points_list]

    # # weight point
    # print(f"clustering_useVirtual(points_list, clustering_radius, associateVirtual=False, withWeight=True): {clustering_useVirtual(points_list, clustering_radius, associateVirtual=False, withWeight=False)}")
    # weightPoints = [point for points in clustering_useVirtual(points_list, clustering_radius, associateVirtual=False) for point in points]
    # print(f"weightPoint: {weightPoints}")
    # axes.scatter(*list(zip(*weightPoints)), c=weight_color, alpha=weight_alpha, label=weight_label)
    # [axes.add_patch(matplotlib.pyplot.Circle(weightPoint, weight_radius, edgecolor=weight_radius_color, facecolor=weight_radius_color, alpha=weight_radius_alpha)) for weightPoint in weightPoints]
    
    # virtual point
    # print(f"clustering_useVirtual(points_list, clustering_radius, associateVirtual=True, withWeight=True): {clustering_useVirtual(points_list, clustering_radius, associateVirtual=True, withWeight=True)}")
    virtualPoints = [point for points in clustering_useVirtual(points_list, clustering_radius, associateVirtual=True) for point in points]
    # print(f"virtualPoints: {virtualPoints}")
    axes.scatter(*list(zip(*virtualPoints)), c=virtual_color, alpha=virtual_alpha, label=virtual_label)
    [axes.add_patch(matplotlib.pyplot.Circle(virtualPoint, virtual_radius, edgecolor=virtual_radius_color, facecolor=virtual_radius_color, alpha=virtual_radius_alpha)) for virtualPoint in virtualPoints]

    # clustering without virtualPoint
    axes.scatter(*list(zip(*clustering_useSource_centers(points_list, clustering_radius))), c="r", label="clustering center without virtualPoint")
    # clustering with virtualPoint
    axes.scatter(*list(zip(*clustering_useVirtual_centers(points_list, clustering_radius, associateVirtual=True, withWeight=True))), c="g", label="clustering center with virtualPoint")

    axes.grid(True)
    axes.legend()

    print("matplotlib.pyplot.show()")
    matplotlib.pyplot.show()

  numpy.random.seed(2023)
  dimension = 2
  size_min  = -15
  size_max  = -size_min

  noise_size = 10

  cores_number = 2
  core_size     = 5
  core_distance = 5

  clustering_label = "clustering"
  clustering_radius = 5

  points_list: list[tuple] = testingData_listTuple_generator(dimension, noise_size, cores_number, core_size, core_distance, size_min, size_max)
  
  Demo_All_V1(points_list, clustering_radius)
  Demo_MatplotlibPlot_V1(points_list, clustering_radius, size_min-core_distance, size_max+core_distance)

# %%
