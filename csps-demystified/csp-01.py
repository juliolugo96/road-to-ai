import pygame
 
WIDTH, HEIGHT = 540, 540
GRID_SIZE = 9
CELL_SIZE = WIDTH // GRID_SIZE
 
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CSP Sudoku Solver")
font = pygame.font.SysFont("Arial", 30)
 
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]
 
def draw_board():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
    screen.fill((255, 255, 255))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if board[i][j] != 0:
                text = font.render(str(board[i][j]), True, (0, 0, 0))
                screen.blit(text, (j * CELL_SIZE + 15, i * CELL_SIZE + 10))
    for i in range(GRID_SIZE + 1):
        thickness = 3 if i % 3 == 0 else 1
        pygame.draw.line(screen, (0, 0, 0),
                         (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE), thickness)
        pygame.draw.line(screen, (0, 0, 0),
                         (i * CELL_SIZE, 0), (i * CELL_SIZE, HEIGHT), thickness)
    pygame.display.update()
 
def is_valid(board, row, col, num):
    # Row and column constraints
    if num in board[row]:
        return False
    if any(board[i][col] == num for i in range(9)):
        return False
    # 3x3 box constraint
    box_x, box_y = col // 3 * 3, row // 3 * 3
    for i in range(box_y, box_y + 3):
        for j in range(box_x, box_x + 3):
            if board[i][j] == num:
                return False
    return True
 
step_counter = [0]

def solve():
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:              # pick a variable
                for num in range(1, 10):      # try a value from its domain
                    if is_valid(board, i, j, num):  # check constraints
                        board[i][j] = num
                        step_counter[0] += 1
                        if step_counter[0] % 50 == 0:
                            draw_board()
                            pygame.time.delay(30)
                        if solve():           # recurse
                            return True
                        board[i][j] = 0       # backtrack
                return False                  # domain exhausted -> dead end
    return True                               # no empty cells -> solved
 
draw_board()
solve()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
