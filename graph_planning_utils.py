from enum import Enum
from queue import PriorityQueue, LifoQueue, Queue
import numpy as np
import numpy.linalg as LA
from scipy.spatial import Voronoi, voronoi_plot_2d
from bresenham import bresenham
import networkx as nx

# Discretize the environment into a grid


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1

    return grid, int(north_min), int(east_min)


def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
        Returns a grid representation of a 2D configuration space
        along with Voronoi graph edges given obstacle data and the
        drone's altitude.
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))
    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))
    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))
    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid
    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])

    # Define a list to hold Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(north - d_north - safety_distance - north_min_center),
                int(north + d_north + safety_distance - north_min_center),
                int(east - d_east - safety_distance - east_min_center),
                int(east + d_east + safety_distance - east_min_center),
            ]
            grid[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3]] = 1

            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # TODO: create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)
    # TODO: check each edge from graph.ridge_vertices for collision
    edges = []
    for edge in graph.ridge_vertices:
        point1 = graph.vertices[edge[0]]
        point2 = graph.vertices[edge[1]]
        cells = list(bresenham(int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1])))
        infeasible = False

        for cell in cells:
            if np.amin(cell) < 0 or cell[0] >= grid.shape[0] or cell[1] >= grid.shape[1]:
                infeasible = True
                break
            if grid[cell[0], cell[1]] == 1:
                infeasible = True
                break

        if infeasible == False:
            point1 = (point1[0], point1[1])
            point2 = (point2[0], point2[1])
            edges.append((point1, point2))

    return grid, edges


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions

## Since, the default center of the grid which is north and east offset (316, 345) is not necessarily a node in the
## graph we will need to readjust these offsets for both start and goal nodes by finding the nearest nodes

def nearestNodes(graph, targetNode):
    nearestNode = None
    smallestDistance = 1e+10
    for node in graph.nodes:
        distance = LA.norm(np.array(node) - np.array(targetNode))
        if distance < smallestDistance:
            nearestNode = node
            smallestDistance = distance
    return nearestNode


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def bfs(graph, heuristic, start, goal):

    path = []
    queue = Queue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]

        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                branch_cost = current_cost + cost
                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((branch_cost, next_node))
                    branch[next_node] = (branch_cost, current_node)
                if next_node == goal:
                    print('Found a path.')
                    found = True
                    break
        if found:
            break

    path = []
    path_cost = 0
    if found:
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])

    return path[::-1], path_cost