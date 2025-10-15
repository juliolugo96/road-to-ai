import pygame, sys, heapq, random
"""
Pac-Man AI Demo using Pygame.

This script implements a simple Pac-Man game where the player controls Pac-Man and three ghosts chase him using different pathfinding algorithms: BFS (Breadth-First Search), DFS (Depth-First Search), and A* (A-star). The game map is defined as an ASCII grid of walls and open spaces.

Modules:
  - pygame: Graphics and input handling.
  - sys: System exit.
  - heapq: Priority queue for A*.
  - random: Random ghost placement and DFS neighbor order.
  - collections.deque: Queue for BFS.

Constants:
  TILE: Size of each tile in pixels.
  FPS: Frames per second.
  GHOST_MOVE_FREQ: How often ghosts move (lower = faster).
  RAW_MAP: ASCII representation of the game map.

Functions:
  - normalize_map(raw_map): Pads map rows to equal length.
  - find(c): Finds the position of character 'c' in the grid.
  - neighbors(x, y): Yields valid neighboring positions from (x, y).
  - reconstruct(came, goal): Reconstructs path from start to goal.
  - bfs(start, goal): Breadth-first search.
  - dfs(start, goal): Depth-first search (randomized neighbor order).
  - astar(start, goal): A* search using Manhattan distance.

Classes:
  - Ghost: Represents a ghost entity with a name, pathfinding algorithm, color, and position.
    - move(target): Moves ghost one step toward target using its algorithm.

Game Loop:
  - Handles player movement via arrow keys.
  - Moves ghosts toward Pac-Man at set intervals.
  - Draws the map, Pac-Man, and ghosts.
  - Ends the game if a ghost catches Pac-Man.

Usage:
  Run the script to start the game. Control Pac-Man with arrow keys. Avoid the ghosts!
"""
from collections import deque
from enum import Enum

# === CONFIG ===
TILE = 24
FPS = 30
GHOST_MOVE_FREQ = 2  # lower = faster ghosts

RAW_MAP = [
    "#############################################################",
    "#P......................#.....................#.............#",
    "#.#####..###########....#....###########..#####....#####....#",
    "#.#.....#.........#.....#.....#.........#.....#....#...#....#",
    "#.#.###.#.#######.#.#########.#.#######.#.###.#.##.#.#.#.###",
    "#.#.#...#.#.....#.#...........#.#.....#.#.#...#.#..#.#.#...#",
    "#.#.#.###.#.###.#.#####...###.#.#.###.#.#.#.###.#.##.#.###.#",
    "#.#.#.....#.#...#...#.#...#...#.#.#...#.#.#...#.#..#.#.#...#",
    "#.#.#######.#.#####.#.#.#####.#.#.#.###.#.###.#.##.#.#.#.###",
    "#.#.........#.....#.#.#.#.....#.#.#...#.#.....#.#..#.#.#...#",
    "#.###############.#.#.#.#.###.#.#.###.#.#######.#.##.#.###.#",
    "#.................#.#.#.#.#...#.#.....#.........#....#.....#",
    "#####.###########.###.#.#.#.###.###########.###########.###",
    "#.....#.........#.....#.#.#.#.............#.........#......#",
    "#.###.#.#######.#####.#.#.#.#.###########.###.###.###.####.#",
    "#...#.#.#.....#.....#.#.#.#.#.#.........#.#...#.#...#......#",
    "###.#.#.#.###.#####.#.#.#.#.#.#######.#.#.#.###.#.#######.#",
    "#...#.#.#.#...#.....#.#.#.#.#.....#...#.#.#.....#.#.....#.#",
    "#.###.#.#.#.###.#####.#.#.#.###.###.#.#.#.#######.#.###.#.#",
    "#.....#.#.#.....#.....#.#.#.#...#...#.#.#.#.....#.#...#.#.#",
    "#######.#.#######.#####.#.#.#.###.###.#.#.#.###.#.#####.#.#",
    "#.........#.....#.....#.#.#.#...#...#.#.#.#.#...#.....#.#.#",
    "#.#########.###.#####.#.#.#.###.###.#.#.#.#.#.#######.#.#.#",
    "#...........#...#.....#.#.#.#...#...#.#.#.#.#.......#.#.#.#",
    "###########.#.###.#####.#.#.#.###.###.#.#.#.#######.#.#.#.#",
    "#...........#.....#.....#.#.#.#...#...#.#.#.#.....#.#.#.#.#",
    "#.###########.#####.#####.#.#.#.###.###.#.#.#.###.#.#.#.#.#",
    "#...................#.....#...#.....#...........#.....#...#",
    "#############################################################"
]

# === NORMALIZE MAP ===
def normalize_map(raw_map):
    """
    Normalizes a map represented as a list of strings by padding each line with '#' characters
    so that all lines have the same length as the longest line.

    Args:
      raw_map (list of str): The input map, where each string represents a row.

    Returns:
      list of str: The normalized map with all rows padded to equal length.
    """
    max_length = max(len(line) for line in raw_map)
    return [line.ljust(max_length, '#') for line in raw_map]

grid = normalize_map(RAW_MAP)
H, W = len(grid), len(grid[0])

class Direction(Enum):
  DOWN = (1, 0)
  UP = (-1, 0)
  RIGHT = (0, 1)
  LEFT = (0, -1)

dirs = [d.value for d in Direction]

def find(c):
    """
    Searches the grid for the first occurrence of the specified character.

    Args:
      c (str): The character to search for in the grid.

    Returns:
      tuple: A tuple (i, j) representing the row and column indices where the character is found.
      None: If the character is not found in the grid.

    Note:
      Assumes that the global variables `grid`, `H`, and `W` are defined and represent
      the grid (as a 2D list), its height, and its width, respectively.
    """
    for i in range(H):
        for j in range(W):
            if grid[i][j]==c:
                return (i,j)
    return None

def neighbors(x,y):
    """
    Yields the valid neighboring positions (nx, ny) for a given cell (x, y) in the grid.

    A neighbor is considered valid if:
    - It is within the bounds of the grid (0 <= nx < H and 0 <= ny < W).
    - The cell at (nx, ny) is not a wall (i.e., grid[nx][ny] != "#").

    Args:
      x (int): The row index of the current cell.
      y (int): The column index of the current cell.

    Yields:
      Tuple[int, int]: The coordinates (nx, ny) of each valid neighboring cell.
    """
    for dx,dy in dirs:
        nx,ny=x+dx,y+dy
        if 0<=nx<H and 0<=ny<W and grid[nx][ny]!="#":
            yield nx,ny

def reconstruct(came, goal):
    """
    Reconstructs the path from the start node to the goal node using the 'came' dictionary.

    Args:
      came (dict): A mapping from each node to its predecessor node.
      goal: The goal node for which the path should be reconstructed.

    Returns:
      list: A list of nodes representing the path from the start node to the goal node.
          Returns an empty list if the goal node is not reachable.
    """
    if goal not in came: return []
    path=[]; cur=goal
    while cur: path.append(cur); cur=came[cur]
    return path[::-1]

# === SEARCH ===
def bfs(start,goal):
    """
    Performs Breadth-First Search (BFS) to find a path from the start node to the goal node.

    Args:
      start: The starting node, typically represented as a tuple (e.g., coordinates).
      goal: The goal node to reach, in the same format as start.

    Returns:
      A list representing the reconstructed path from start to goal, or None if no path exists.

    Note:
      This function assumes the existence of a `neighbors` function that returns adjacent nodes,
      and a `reconstruct` function that rebuilds the path from the `came` dictionary.
    """
    q=deque([start]); came={start:None}
    while q:
        cur = q.popleft()
        if cur == goal: break
        for n in neighbors(*cur):
            if n not in came:
                came[n]=cur; q.append(n)
    return reconstruct(came,goal)

def dfs(start, goal):
    """
    Performs a depth-first search (DFS) from the start node to the goal node.

    Args:
      start (tuple): The starting node coordinates.
      goal (tuple): The goal node coordinates.

    Returns:
      list: A list of nodes representing the path from start to goal, reconstructed using the 'came' dictionary.

    Notes:
      - The function assumes the existence of a 'neighbors' function that returns adjacent nodes for a given node.
      - The path is reconstructed using a 'reconstruct' function.
      - Neighbor nodes are shuffled to randomize the search order.
    """
    stack = [start]
    came = {start: None}
    while stack:
        cur = stack.pop()
        if cur == goal:
            break
        neighs = list(neighbors(*cur))
        random.shuffle(neighs)
        for n in neighs:
            if n not in came:
                came[n] = cur
                stack.append(n)
    return reconstruct(came, goal)

def astar(start, goal):
    """
    Finds the shortest path from the start position to the goal position using the A* search algorithm.
    Args:
      start (tuple): The starting position as a tuple of coordinates (x, y).
      goal (tuple): The goal position as a tuple of coordinates (x, y).
    Returns:
      list: A list of positions (tuples) representing the path from start to goal, or an empty list if no path is found.
    Notes:
      - The function uses Manhattan distance as the heuristic.
      - Neighboring nodes are ordered by proximity to the goal to introduce a bias.
      - Assumes the existence of `neighbors` and `reconstruct` functions, and the `heapq` module is imported.
    """
    
    def heuristic(a,b):
        """
        Calculates the Manhattan distance between two points.

        Args:
          a (tuple): The (x, y) coordinates of the first point.
          b (tuple): The (x, y) coordinates of the second point.

        Returns:
          int: The Manhattan distance between point a and point b.
        """
        return abs(a[0]-b[0])+abs(a[1]-b[1])

    f = [(0, start)]
    came = {start: None}
    cost = {start: 0}

    while f:
        _, cur = heapq.heappop(f)
        if cur == goal:
            break

        # Order neighbors by heuristic proximity to goal (bias)
        neighs = sorted(neighbors(*cur), key=lambda n: heuristic(n, goal))
        for n in neighs:
            # add a small bias: if moving toward goal, it's slightly cheaper
            step_cost = 1 + 0.001 * heuristic(n, goal)
            new_cost = cost[cur] + step_cost
            if n not in cost or new_cost < cost[n]:
                cost[n] = new_cost
                priority = new_cost + heuristic(n, goal)
                heapq.heappush(f, (priority, n))
                came[n] = cur

    return reconstruct(came, goal)


# === ENTITIES ===
class Ghost:
    """
    Represents a ghost character in the Pac-Man game.

    Attributes:
      name (str): The name of the ghost.
      algo (str): The pathfinding algorithm used by the ghost ('bfs', 'dfs', or 'astar').
      color (str): The color of the ghost.
      pos (Any): The current position of the ghost on the game board.

    Methods:
      move(target):
        Moves the ghost towards the target position using the specified pathfinding algorithm.
        If the ghost's position is not set, the method does nothing.
        Args:
          target (Any): The target position to move towards.
    """
    def __init__(self, name, algo, color):
        self.name=name
        self.algo=algo
        self.color=color
        self.pos=None

    def move(self,target):
        if not self.pos: return
        path={"bfs":bfs,"dfs":dfs,"astar":astar}[self.algo](self.pos,target)
        if len(path)>1:
            self.pos=path[1]

# === INIT ===
def init_pygame():
    """
    Initializes the Pygame library and sets up the game window and clock.

    Returns:
      tuple: A tuple containing the Pygame screen surface and clock object.
    """
    pygame.init()
    screen=pygame.display.set_mode((W*24, H*24))
    clock=pygame.time.Clock()
    return screen, clock

def setup():
    """
    Sets up the initial game state, including loading the map, initializing entities, and preparing the Pygame environment.

    Returns:
      tuple: A tuple containing the Pygame screen surface, clock object
    """
    screen, clock = init_pygame()

    pac=find('P')
    opens=[(i,j) for i in range(H) for j in range(W) if grid[i][j]=='.']

    ghosts=[
        Ghost('Blinky','bfs',(255,0,0)),     # BFS - red
        Ghost('Inky','dfs',(0,255,0)),       # DFS - green
        Ghost('Clyde','astar',(255,192,0))   # A* - yellow
    ]

    for g in ghosts:
        g.pos=random.choice(opens)

    return screen, clock, pac, ghosts

# === DRAW ===
def draw(screen, pac, ghosts):
    """
    Renders the current state of the Pac-Man game onto the screen.

    - Fills the screen with a black background.
    - Draws the game grid, coloring walls and empty spaces differently.
    - Draws Pac-Man as a yellow circle at its current position.
    - Draws all ghosts as colored circles at their respective positions.
    - Updates the display to reflect the changes.

    Assumes the existence of global variables:
      - screen: The pygame display surface.
      - H, W: Grid height and width.
      - grid: 2D list representing the game map.
      - pac: Tuple (row, col) for Pac-Man's position.
      - ghosts: List of ghost objects, each with 'pos' and 'color' attributes.
    """
    screen.fill((0,0,0))
    for i in range(H):
        for j in range(W):
            color=(30,30,70) if grid[i][j]=='#' else (10,10,10)
            pygame.draw.rect(screen,color,(j*24,i*24,24,24))
    px,py=pac
    pygame.draw.circle(screen,(255,255,0),(py*24+12,px*24+12),10)
    for g in ghosts:
        gx,gy=g.pos
        pygame.draw.circle(screen,g.color,(gy*24+12,gx*24+12),10)
    pygame.display.flip()

def print_instructions():
    """
    Prints the game instructions to the console.
    """
    print("=== PAC-MAN AI DEMO ===")
    print("Use arrow keys to move Pac-Man.")
    print("Avoid the ghosts (Blinky, Inky, Clyde)!")
    print("Blinky uses BFS (red), Inky uses DFS (green), Clyde uses A* (yellow).")
    print("Press the window's close button to exit.")

def play_game():
    """
    Main game
    Initializes the game, handles the game loop, processes player input, updates game state,
    and renders the game until the player exits or is caught by a ghost.
    """
    screen, clock, pac, ghosts = setup()
    print_instructions()
    running = True
    frame = 0
    while running:
        clock.tick(FPS)
        frame += 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # Pacman move
        keys = pygame.key.get_pressed()
        move = None
        if keys[pygame.K_UP]: move = Direction.UP.value
        elif keys[pygame.K_DOWN]: move = Direction.DOWN.value
        elif keys[pygame.K_LEFT]: move = Direction.LEFT.value
        elif keys[pygame.K_RIGHT]: move = Direction.RIGHT.value
        if move:
            x, y = pac
            nx, ny = x + move[0], y + move[1]
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                pac = (nx, ny)

        # 👻 Ghost moves — now steady and visible
        if frame % (FPS // GHOST_MOVE_FREQ) == 0:
            for g in ghosts:
                g.move(pac)
                if g.pos == pac:
                    print(f"💀 You got caught by {g.name} ({g.algo.upper()})!")
                    running = False
                    break

        draw(screen, pac, ghosts)


    pygame.quit()
    sys.exit()

play_game()
