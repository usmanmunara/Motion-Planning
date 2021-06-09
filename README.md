# FCND - 3D Motion Planning

![Quad Image](./misc/enroute.png)

**Worked on [FCND Motion Planning](https://github.com/udacity/FCND-Motion-Planning) from UDACITY implementing the following:**

## Implementing the depth-first search algorithm

You are expected to write a depth-first search algorithm in planning_utils.py named dfs(grid, h, start,
goal) to help the drone plan routes. Pseudocode for the depth-first search algorithm can be found in
the lecture slides.

## Implementing the iterative deepening A\* search algorithm

You are expected to write an iterative deepening A* search algorithm in planning_utils.py named
iterative_astar(grid, h, start, goal) to help the drone plan routes. Procedure for the iterative
deepening A* search algorithm can be found in the lecture slides.

## Implementing the uniform cost search algorithm

You are expected to write a uniform cost search algorithm in planning_utils.py named ucs(grid, h,
start, goal) to help the drone plan routes. Procedure for the uniform cost search algorithm can be
found in the lecture slides. Note that you should first of all design and implement the cost function.

## Implementing different heuristics for A*

In the current A* version in planning_utils.py, the Manhatten distance is used as the heuristic. You
are encouraged to propose one valid heuristic, implement that, and see how the planned routes
change.

## Implementing the breadth-first search algorithm with graph search

You are expected to shift the representation of the state space from grids in the previous questions
to a Voronoi graph.

## Implementing the A\* search for traversing 3 fixed points

You are required to set three fixed points in motion_planning.py which the drone has to traverse before reaching to the destination. Building on this, you re-implement the A\* search algorithm in order to go through the 3 points
