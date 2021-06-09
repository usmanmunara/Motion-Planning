# Muhammad Usman FAROOQ - 55301764

from enum import Enum
from queue import PriorityQueue, LifoQueue, Queue
import numpy as np
from heapdict import heapdict
from scipy.spatial.distance import cityblock, minkowski
from scipy.spatial import Voronoi, voronoi_plot_2d
from bresenham import bresenham
import networkx as nx
import numpy.linalg as LA
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
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

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    # print("next node", next_node)
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


def heuristic(position, goal_position):
    # Euclidean
    return np.linalg.norm(np.array(position) - np.array(goal_position))

# Question 4


def manhattanHeuristic(position, goal_position):
    # Use manhattan distance instead of euclidean as heuristic
    return cityblock(np.array(position), np.array(goal_position))


def minkowskiHeurisitc(position, goal_position):
    # Alternatively one can just use minkowski heurisitc with an arbitrary p
    p = 3
    return minkowski(np.array(position), np.array(goal_position), p)

# Question 1


def dfs(grid, h, start, goal):
    path = []
    path_cost = 0
    queue = LifoQueue()
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

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost

                if next_node not in visited:
                    # print("next node", next_node)
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost

# Question 2


def iterative_astar(grid, h, start, goal):
    '''
     For iterative deepening A* the cutoff occurs when the cuttoff value
     at each iteration > the cutoff value on the previous iteration

     depth_limited_search is a recursive

     cutoff = f(g + h) = path cost + heuristic cost
    '''

    threshold = h(start, goal)
    visited = []
    path = []

    for depth in range(sys.maxsize):
        cost, path, visited = depth_limited_search(grid, h, start, goal, 0, threshold, path, visited)
        if cost == float("inf"):
            print('**********************')
            print('Failed to find a path!')
            print('**********************')
        elif cost < 0:
            print("Found a path.")
            return path[::-1], cost
        else:
            threshold = cost
            path = []
            visited = []


def depth_limited_search(grid, h, current_node, goal, cost, threshold, path, visited):
    '''
    This is a recursive depth first search with cuttoff.
     cutoff = f(g + h) = path cost + heuristic cost
     implementation is similar to that provided in AIMA book for
     depth first search with limit as cutoff -
     https://github.com/aimacode/aima-python/blob/master/search.py
    '''

    visited.append(current_node)
    cutoff = cost + h(current_node, goal)
    if current_node == goal:
        path.append(goal)
        return -cost, path, visited
    if cutoff > threshold:
        return cutoff, path, visited

    min_cost = float("inf")
    for action in valid_actions(grid, current_node):
        da = action.delta
        next_node = (current_node[0] + da[0], current_node[1] + da[1])
        if next_node not in visited:
            cuttoff_cost = cost + action.cost
            child_cost, path, visited = depth_limited_search(
                grid, h, next_node, goal, cuttoff_cost, threshold, path, visited)
            if child_cost < 0:
                path.append(current_node)
                return child_cost, path, visited
            elif child_cost < min_cost:
                min_cost = child_cost
    return min_cost, path, visited


# Question 3
def ucs(grid, h, start, goal):
    '''

    Since we can't access and edit the queue provided by python's queue library
    We will use a heap dictionary to keep track of the frontier of UCS similar to
    lecture and AIMA
    '''
    path = []
    path_cost = 0.0
    frontier = heapdict()
    frontier[start] = 0
    visited = set(start)
    branch = {}
    found = False

    while frontier:
        # Choose the lowest cost node
        item = frontier.popitem()

        current_cost = item[1]
        current_node = item[0]

        # Apply goal test when chosen for expansion
        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    frontier[next_node] = queue_cost
                elif (frontier.get(next_node, "does not exist") != "does not exist") and (branch_cost < frontier[next_node]):
                    frontier[next_node] = branch_cost
                    branch[next_node] = (branch_cost, current_node, action)

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


# Question 5
'''
Resources used:
http://orensalzman.com/docs/roadmaps.pdf
http://www.cs.cmu.edu/afs/andrew/course/15/381-f08/www/lectures/motionplanning.pdf (Specially slide 6 and 7)
'''


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

    # location of obstacle centres
    graph = Voronoi(points)
    # check each edge from graph.ridge_vertices for collision
    edges = []
    for edge in graph.ridge_vertices:
        point1 = graph.vertices[edge[0]]
        point2 = graph.vertices[edge[1]]

        # Use bresenham's line algorithm for collision detection
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


def visualizeGraph(grid, edges):
    print('Found %5d edges' % len(edges))
    plt.imshow(grid, origin='lower', cmap='Greys')


def createGraph(edges):
    """
    After extracting the edges from the voronoi diagram Create a graph
    """
    G = nx.Graph()
    for e in edges:
        point1 = e[0]
        point2 = e[1]
        distance = LA.norm(np.array(point2) - np.array(point1))
        G.add_edge(point1, point2, weight=distance)
    return G


def nearestNodes(graph, targetNode):
    '''
    Since, the default center of the grid which is north and east offset (316, 345) is not necessarily a node in the
    graph we will need to readjust these offsets for both start and goal nodes by finding the nearest nodes
    '''

    nearestNode = None
    smallestDistance = float('inf')
    for node in graph.nodes:
        distance = LA.norm(np.array(node) - np.array(targetNode))
        if distance < smallestDistance:
            nearestNode = node
            smallestDistance = distance
    return nearestNode


def bfs(graph, h, start, goal):

    path = []
    queue = Queue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        current_cost = item[0]

        for next_node in graph[current_node]:
            cost = graph.edges[current_node, next_node]['weight']
            branch_cost = current_cost + cost
            if next_node not in visited:
                visited.add(next_node)
                queue.put((branch_cost, next_node))
                branch[next_node] = (branch_cost, current_node)
            # Apply the goal test when generated
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
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost

# Question 6


def fixedPointsAStarTraversal(grid, h, start, goal, point1, point2, point3):
    '''

    Originally, we have one start(P0) and one goal point(PF) where we do A* on (P0, PF). Since we need three fixed points, we can break the original path planning into a multi-step path planning of
    - Run A* with start and goal of (P0, P1), then
    - Run A* with start and goal of (P1, P2), then
    - Run A* with start and goal of (P2, P3), then finally
    - Run A* with start and goal of (P3, PF)

    However here we calculate distances between the three alloted points first
    to make it optimal for the A* traversal because if we go P1, P2, P3 in order
    provided then these traversals will be costly sometimes
    '''
    distance1 = h(start, point1) + h(point1, goal)
    distance2 = h(start, point2) + h(point2, goal)
    distance3 = h(start, point3) + h(point3, goal)
    if distance1 >= distance2 and distance1 >= distance3:
        path1, cost1 = a_star(grid, h, start, point1)
        if distance2 >= distance3:
            path2, cost2 = a_star(grid, h, point1, point2)
            path3, cost3 = a_star(grid, h, point2, point3)
            path4, cost4 = a_star(grid, h, point3, goal)
        else:
            path2, cost2 = a_star(grid, h, point1, point3)
            path3, cost3 = a_star(grid, h, point3, point2)
            path4, cost4 = a_star(grid, h, point2, goal)
    elif distance2 >= distance3 and distance2 >= distance1:
        path1, cost1 = a_star(grid, h, start, point2)
        if distance1 >= distance3:
            path2, cost2 = a_star(grid, h, point2, point1)
            path3, cost3 = a_star(grid, h, point1, point3)
            path4, cost4 = a_star(grid, h, point3, goal)
        else:
            path2, cost2 = a_star(grid, h, point2, point3)
            path3, cost3 = a_star(grid, h, point3, point1)
            path4, cost4 = a_star(grid, h, point1, goal)
    else:
        path1, cost1 = a_star(grid, h, start, point3)
        if distance2 >= distance1:
            path2, cost2 = a_star(grid, h, point3, point2)
            path3, cost3 = a_star(grid, h, point2, point1)
            path4, cost4 = a_star(grid, h, point1, goal)
        else:
            path2, cost2 = a_star(grid, h, point3, point1)
            path3, cost3 = a_star(grid, h, point1, point2)
            path4, cost4 = a_star(grid, h, point2, goal)

    path = path1 + path2 + path3 + path4
    cost = cost1 + cost2 + cost3 + cost4
    return path, cost

def alternative_fixedPointsAStarTraversal(grid, h, start, goal, point1, point2, point3):
    '''

    Originally, we have one start(P0) and one goal point(PF) where we do A* on (P0, PF). Since we need three fixed points, we can break the original path planning into a multi-step path planning of
    - Run A* with start and goal of (P0, P1), then
    - Run A* with start and goal of (P1, P2), then
    - Run A* with start and goal of (P2, P3), then finally
    - Run A* with start and goal of (P3, PF)

   This is an alternative version where there is no optimality applied for the three points.
    '''
    path1, cost1 =  a_star(grid, h, start, point1)
    path2, cost2 = a_star(grid, h, point1, point2)
    path3, cost3 = a_star(grid, h, point2, point3)
    path4, cost4 = a_star(grid, h, point3, goal)
    path = path1 + path2 + path3 + path4
    cost = cost1 + cost2 + cost3 + cost4
    return path, cost