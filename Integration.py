import numpy # numpy-1.26.0

# Calculate the distance between two points
distance = lambda point1, point2: float(numpy.linalg.norm(numpy.array(point1) - numpy.array(point2)))
# Calculate the center point of a cluster
center = lambda cluster: tuple(numpy.array([sum(vL) for vL in numpy.transpose(numpy.array(list(cluster)))]) / len(cluster))

def integration_V1_universal(points: list, rule, point=lambda point: point) -> list[tuple]:
  """merge adjacent points into one point

    1. Iterate through each point in the input list
    2. Check if the point belongs to any existing cluster based on the rule
    3. If the point does not belong to any cluster, create a new cluster
    4. If the point belongs to multiple clusters, merge those clusters
    5. Calculate the center point as the average of its points

  Args:
      points (list): Coordinate points to be processed
      rule (_type_): lambda
      point (_type_, optional): method to get coordinates. Defaults to `lambda point: point`.

  Returns:
      list[tuple]: Merged coordinates
  """
  clusters: list[set] = []
  
  for i in range(len(points)):
    for j in range(i+1, len(points)):
      # Join or add a new cluster
      if rule(points[i], points[j]):
        clustered = False
        newCluster = {point(points[i]), point(points[j])}
        # Join cluster
        for cluster in clusters:
          if not cluster.isdisjoint(newCluster):
            cluster |= newCluster
            clustered = True
            break  # Exit the loop after merging the cluster
        # Add a new cluster
        if not clustered:
          clusters.append(newCluster)

  # Merge associated clusters
  i = 0
  while i < len(clusters):
    j = i + 1
    while j < len(clusters):
      if clusters[i].isdisjoint(clusters[j]):
        j += 1
      else:
        clusters[i] |= clusters.pop(j)
    i += 1

  # Calculate the center point for each cluster
  return [center(cluster) for cluster in clusters if cluster]

def integration_V2_1(points: list[tuple], limit: float | int, scaling: float | int) -> list[tuple]:
  """Integration and filtering of detection points

    1. Generate weighted virtual points
      1.1. Position: midpoint between two points
      1.2. Weight: log⁡(1/(𝑑/𝑟))
        1.2.1. 𝑑 : distance between two points
        1.2.2. 𝑟: distance limitation coefficient
    2. Generate virtual point clusters using existing virtual points as reference targets
      2.1. Point cluster: The weight of the generated virtual points is greater than 0, and the distance between virtual points is less than α𝑟
        2.1.1. α	distance limit scaling factor
    3. Merge virtual point cluster to generate merged points

  Args:
      points (list[tuple]): Coordinate points to be processed
      limit (float | int): distance limitation coefficient
      scaling (float | int): distance limit scaling factor

  Returns:
      list[tuple]: Merged coordinates
  """
  class VirtualPoint:
    def __init__(self, point1: tuple, point2: tuple, limit: float):
      self.sources = (numpy.array(point1), numpy.array(point2))
      # 1.1. Position: midpoint between two points
      self.point = tuple((self.sources[0] + self.sources[1])/2)
      # 1.2. Weight: log⁡(1/(𝑑/𝑟))
      #   1.2.1. 𝑑 : distance between two points
      #   1.2.2. 𝑟: distance limitation coefficient
      self.weight = numpy.log(1/(numpy.linalg.norm(self.sources[0]- self.sources[1])/limit))
  # 1. Generate weighted virtual points # NOTE: not full
  virtualPoints: list[VirtualPoint] = [VirtualPoint(points[i], points[j], limit) for i in range(len(points)) for j in range(i+1, len(points))]
  # 2. Generate virtual point clusters using existing virtual points as reference targets
  # 3. Merge virtual point cluster to generate merged points
  virtualClusterCenters = integration_V1_universal(virtualPoints, lambda point1, point2: point1.weight>0 and point2.weight>0 and distance(point1.point, point2.point) < scaling*limit, lambda virtualPoint: virtualPoint.point)
  return virtualClusterCenters

def integration_V2_2(points: list[tuple], limit: float | int, scaling: float | int) -> list[tuple]:
  """Integration and filtering of detection points

    1. Generate weighted virtual points
      1.1. Position: midpoint between two points
      1.2. Weight: log⁡(1/(𝑑/𝑟))
        1.2.1. 𝑑 : distance between two points
        1.2.2. 𝑟: distance limitation coefficient
    2. Generate point clusters using existing virtual points as reference targets
      2.1. Point clusters: The weight of the generated virtual points is greater than 0, and the distance between virtual points is less than α𝑟
        2.1.1. α	distance limit scaling factor
    3. Merge point clusters to generate merged points

  Args:
      points (list[tuple]): Coordinate points to be processed
      limit (float | int): distance limitation coefficient
      scaling (float | int): distance limit scaling factor

  Returns:
      list[tuple]: Merged coordinates
  """
  class VirtualPoint:
    def __init__(self, point1: tuple, point2: tuple, limit: float):
      self.sources = (numpy.array(point1), numpy.array(point2))
      # 1.1. Position: midpoint between two points
      self.point = tuple((self.sources[0] + self.sources[1])/2)
      # 1.2. Weight: log⁡(1/(𝑑/𝑟))
      #   1.2.1. 𝑑 : distance between two points
      #   1.2.2. 𝑟: distance limitation coefficient
      self.weight = numpy.log(1/(numpy.linalg.norm(self.sources[0]- self.sources[1])/limit))
      self.sources = (tuple(self.sources[0]), tuple(self.sources[1]))
  # 1. Generate weighted virtual points
  virtualPoints: list[VirtualPoint] = [VirtualPoint(points[i], points[j], limit) for i in range(len(points)) for j in range(i+1, len(points))]
  # 2. Generate point clusters using existing virtual points as reference targets
  clusters: list[set] = []
  for i in range(len(virtualPoints)):
    for j in range(i+1, len(virtualPoints)):

      # Join or add a new cluster
      if virtualPoints[i].weight>0 and virtualPoints[j].weight>0 and distance(virtualPoints[i].point, virtualPoints[j].point) < scaling*limit:
        clustered = False
        newCluster = set(virtualPoints[i].sources) | set(virtualPoints[j].sources)
        # Join cluster
        for cluster in clusters:
          if cluster.isdisjoint(newCluster):
            cluster |= newCluster
            clustered = True
        # Add a new cluster
        if not clustered:
          clusters.append(newCluster)

      # Merge associated clusters
      i = 0
      while i < len(clusters):
        j = i + 1
        while j < len(clusters):
          if clusters[i].isdisjoint(clusters[j]):
            j += 1
          else:
            clusters[i] |= clusters.pop(j)
        i += 1
  # 3. Merge point clusters to generate merged points
  return [center(cluster) for cluster in clusters]

def integration_V2_3(points: list[tuple], limit: float | int, scaling: float | int) -> list[tuple]:
  """Integration and filtering of detection points

    1. Generate weighted virtual points
      1.1. Position: midpoint between two points
      1.2. Weight: log⁡(1/(𝑑/𝑟))
        1.2.1. 𝑑 : distance between two points
        1.2.2. 𝑟 : distance limitation coefficient
    2. Generate point clusters using existing virtual points as reference targets
      2.1. Point cluster: The weight of the generated virtual points is greater than 0, and the distance between virtual points is less than α𝑟
        2.1.1. α	distance limit scaling factor
    3. The virtual point weights within each point cluster are synthesized into each point weight.
      3.1. Weight: virtual point weight average
    4. Calculate the center of gravity of each point cluster

  Args:
      points (list[tuple]): Coordinate points to be processed
      limit (float | int): distance limitation coefficient
      scaling (float | int): distance limit scaling factor

  Returns:
      list[tuple]: Merged coordinates
  """
  class VirtualPoint:
    def __init__(self, point1: tuple, point2: tuple, limit: float):
      self.sources = (numpy.array(point1), numpy.array(point2))
      # 1.1. Position: midpoint between two points
      self.point = tuple((self.sources[0] + self.sources[1])/2)
      # 1.2. Weight: log⁡(1/(𝑑/𝑟))
      #   1.2.1. 𝑑 : distance between two points
      #   1.2.2. 𝑟: distance limitation coefficient
      self.weight = numpy.log(1/(numpy.linalg.norm(self.sources[0]- self.sources[1])/limit))
      self.sources = (tuple(self.sources[0]), tuple(self.sources[1]))
  # 1. Generate weighted virtual points
  virtualPoints: list[VirtualPoint] = [VirtualPoint(points[i], points[j], limit) for i in range(len(points)) for j in range(i+1, len(points))]
  # 2. Generate point clusters using existing virtual points as reference targets
  clusters: list[set] = []
  for i in range(len(virtualPoints)):
    for j in range(i+1, len(virtualPoints)):

      # Join or add a new cluster
      if virtualPoints[i].weight>0 and virtualPoints[j].weight>0 and distance(virtualPoints[i].point, virtualPoints[j].point) < scaling*limit:
        clustered = False
        newCluster = set(virtualPoints[i].sources) | set(virtualPoints[j].sources)
        # Join cluster
        for cluster in clusters:
          if cluster.isdisjoint(newCluster):
            cluster |= newCluster
            clustered = True
        # Add a new cluster
        if not clustered:
          clusters.append(newCluster)

      # Merge associated clusters
      i = 0
      while i < len(clusters):
        j = i + 1
        while j < len(clusters):
          if clusters[i].isdisjoint(clusters[j]):
            j += 1
          else:
            clusters[i] |= clusters.pop(j)
        i += 1
  # 3. The virtual point weights within each point cluster are synthesized into each point weight.
  weightClusters = []
  for cluster in clusters:
    cluster_list = list(cluster)
    weighted_dict = {}
    for i in range(len(cluster_list)):
      weighted_dict[cluster_list[i]] = sum([VirtualPoint(cluster_list[i], cluster_list[j], limit).weight for j in range(len(cluster)) if i != j]) / len(cluster)
    weightClusters.append(weighted_dict)
  # 4. Calculate the center of gravity of each point cluster
  def calculate_center_of_gravity(weighted_dict):
    total_weight = sum(weighted_dict.values())
    center_coordinates = [0] * len(next(iter(weighted_dict.keys())))
    for point, weight in weighted_dict.items():
      for i, coordinate in enumerate(point):
        center_coordinates[i] += coordinate * (weight / total_weight)
    return tuple(center_coordinates)
  return [calculate_center_of_gravity(wp) for wp in weightClusters]
