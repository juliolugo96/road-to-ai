import math
import time
import pygame
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ------------------------------------------------------------
# Minimax vs Alpha-Beta Pruning Visualizer for Tic-Tac-Toe
# ------------------------------------------------------------
# Controls:
#   SPACE  -> start / restart both games from the same board state
#   R      -> reset and start again
#   ESC    -> quit
#
# What it shows:
#   - Two tic-tac-toe boards side by side
#   - Left board uses plain Minimax
#   - Right board uses Alpha-Beta pruning
#   - Each side animates the search by highlighting explored cells
#   - Each side displays nodes visited, elapsed search time, and chosen move
#
# Notes:
#   - Both AIs play optimally from the same position on each turn
#   - The simulation steps one move at a time, so the audience can watch
#     how both algorithms reason before making their move.
# ------------------------------------------------------------

pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 1400, 760
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Minimax vs Alpha-Beta Pruning — Tic-Tac-Toe Visualizer")
CLOCK = pygame.time.Clock()
FPS = 60

BG = (18, 18, 24)
PANEL = (28, 30, 40)
PANEL_ALT = (34, 37, 50)
GRID = (220, 220, 230)
TEXT = (240, 240, 245)
MUTED = (170, 175, 190)
ACCENT_MINIMAX = (255, 166, 77)
ACCENT_AB = (92, 190, 255)
WIN_LINE = (95, 230, 140)
HILITE = (255, 235, 120)
PRUNED = (255, 110, 110)
X_COLOR = (255, 120, 120)
O_COLOR = (120, 190, 255)

TITLE_FONT = pygame.font.SysFont("arial", 30, bold=True)
SUBTITLE_FONT = pygame.font.SysFont("arial", 22, bold=True)
BODY_FONT = pygame.font.SysFont("arial", 20)
SMALL_FONT = pygame.font.SysFont("arial", 16)
BIG_FONT = pygame.font.SysFont("arial", 36, bold=True)

BOARD_SIZE = 360
CELL_SIZE = BOARD_SIZE // 3
TOP_MARGIN = 180
LEFT_X = 120
RIGHT_X = WIDTH - LEFT_X - BOARD_SIZE
BOARD_Y = TOP_MARGIN

SEARCH_HILITE_MS = 180
MOVE_APPLY_DELAY_MS = 650
END_DELAY_MS = 1200

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def other_player(player: str) -> str:
    return "O" if player == "X" else "X"


def available_moves(board: List[str]) -> List[int]:
    return [i for i, cell in enumerate(board) if cell == ""]


def check_winner(board: List[str]) -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
    for a, b, c in LINES:
        if board[a] != "" and board[a] == board[b] == board[c]:
            return board[a], (a, b, c)
    if all(cell != "" for cell in board):
        return "Draw", None
    return None, None


def evaluate_terminal(board: List[str], ai_player: str, depth: int) -> Optional[int]:
    winner, _ = check_winner(board)
    if winner is None:
        return None
    if winner == "Draw":
        return 0
    if winner == ai_player:
        return 10 - depth
    return depth - 10


@dataclass
class SearchStep:
    move_index: int
    depth: int
    kind: str  # "visit", "return", "prune"
    value: Optional[int] = None
    alpha: Optional[int] = None
    beta: Optional[int] = None


@dataclass
class SearchResult:
    move: int
    score: int
    nodes_visited: int
    elapsed_ms: float
    steps: List[SearchStep]


class MinimaxSolver:
    def choose_move(self, board: List[str], ai_player: str) -> SearchResult:
        steps: List[SearchStep] = []
        nodes_visited = 0
        start = time.perf_counter()

        def minimax(curr_board: List[str], current_player: str, depth: int) -> int:
            nonlocal nodes_visited
            nodes_visited += 1

            terminal = evaluate_terminal(curr_board, ai_player, depth)
            if terminal is not None:
                return terminal

            moves = available_moves(curr_board)
            maximizing = current_player == ai_player

            if maximizing:
                best = -math.inf
                for move in moves:
                    curr_board[move] = current_player
                    score = minimax(curr_board, other_player(current_player), depth + 1)
                    curr_board[move] = ""
                    best = max(best, score)
                return int(best)
            else:
                best = math.inf
                for move in moves:
                    curr_board[move] = current_player
                    score = minimax(curr_board, other_player(current_player), depth + 1)
                    curr_board[move] = ""
                    best = min(best, score)
                return int(best)

        best_score = -math.inf
        best_move = -1
        for move in available_moves(board[:]):
            steps.append(SearchStep(move, 0, "visit"))
            board[move] = ai_player
            score = minimax(board, other_player(ai_player), 1)
            board[move] = ""
            steps.append(SearchStep(move, 0, "return", value=score))
            if score > best_score:
                best_score = score
                best_move = move

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SearchResult(best_move, int(best_score), nodes_visited, elapsed_ms, steps)


class AlphaBetaSolver:
    def choose_move(self, board: List[str], ai_player: str) -> SearchResult:
        steps: List[SearchStep] = []
        nodes_visited = 0
        start = time.perf_counter()

        def alphabeta(curr_board: List[str], current_player: str, depth: int, alpha: int, beta: int) -> int:
            nonlocal nodes_visited
            nodes_visited += 1

            terminal = evaluate_terminal(curr_board, ai_player, depth)
            if terminal is not None:
                return terminal

            moves = available_moves(curr_board)
            maximizing = current_player == ai_player

            if maximizing:
                value = -math.inf
                for move in moves:
                    curr_board[move] = current_player
                    score = alphabeta(curr_board, other_player(current_player), depth + 1, alpha, beta)
                    curr_board[move] = ""
                    value = max(value, score)
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return int(value)
            else:
                value = math.inf
                for move in moves:
                    curr_board[move] = current_player
                    score = alphabeta(curr_board, other_player(current_player), depth + 1, alpha, beta)
                    curr_board[move] = ""
                    value = min(value, score)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return int(value)

        best_score = -math.inf
        best_move = -1
        alpha = -math.inf
        beta = math.inf
        for move in available_moves(board[:]):
            steps.append(SearchStep(move, 0, "visit", alpha=alpha, beta=beta))
            board[move] = ai_player
            score = alphabeta(board, other_player(ai_player), 1, alpha, beta)
            board[move] = ""
            steps.append(SearchStep(move, 0, "return", value=score, alpha=alpha, beta=beta))
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SearchResult(best_move, int(best_score), nodes_visited, elapsed_ms, steps)


@dataclass
class BoardViewState:
    name: str
    accent: Tuple[int, int, int]
    x: int
    board: List[str]
    solver: object
    result: Optional[SearchResult] = None
    winner: Optional[str] = None
    win_line: Optional[Tuple[int, int, int]] = None
    active_step_index: int = 0
    step_timer_ms: int = 0
    searching: bool = True
    move_applied: bool = False
    move_apply_timer_ms: int = 0
    final_move: Optional[int] = None
    status_text: str = "Thinking..."
    last_highlight: Optional[int] = None
    last_return_value: Optional[int] = None
    prune_flash: bool = False

    def reset_for_search(self, source_board: List[str]) -> None:
        self.board = source_board[:]
        self.result = None
        self.winner = None
        self.win_line = None
        self.active_step_index = 0
        self.step_timer_ms = 0
        self.searching = True
        self.move_applied = False
        self.move_apply_timer_ms = 0
        self.final_move = None
        self.status_text = "Thinking..."
        self.last_highlight = None
        self.last_return_value = None
        self.prune_flash = False


def draw_text(text: str, font: pygame.font.Font, color: Tuple[int, int, int], x: int, y: int) -> None:
    surface = font.render(text, True, color)
    SCREEN.blit(surface, (x, y))


def draw_centered_text(text: str, font: pygame.font.Font, color: Tuple[int, int, int], rect: pygame.Rect) -> None:
    surface = font.render(text, True, color)
    SCREEN.blit(surface, surface.get_rect(center=rect.center))


def draw_panel(view: BoardViewState, current_player: str, synced_turn: int, baseline_board: List[str]) -> None:
    panel_rect = pygame.Rect(view.x - 30, 120, BOARD_SIZE + 60, 560)
    pygame.draw.rect(SCREEN, PANEL if view.name == "Minimax" else PANEL_ALT, panel_rect, border_radius=18)
    pygame.draw.rect(SCREEN, view.accent, panel_rect, width=2, border_radius=18)

    draw_text(view.name, BIG_FONT, view.accent, view.x, 132)
    draw_text(f"Current player: {current_player}", BODY_FONT, TEXT, view.x, 175)
    draw_text(f"Turn: {synced_turn + 1}", BODY_FONT, MUTED, view.x + 190, 175)

    board_rect = pygame.Rect(view.x, BOARD_Y, BOARD_SIZE, BOARD_SIZE)
    pygame.draw.rect(SCREEN, (22, 24, 32), board_rect, border_radius=8)

    for i in range(1, 3):
        pygame.draw.line(SCREEN, GRID, (view.x + i * CELL_SIZE, BOARD_Y), (view.x + i * CELL_SIZE, BOARD_Y + BOARD_SIZE), 3)
        pygame.draw.line(SCREEN, GRID, (view.x, BOARD_Y + i * CELL_SIZE), (view.x + BOARD_SIZE, BOARD_Y + i * CELL_SIZE), 3)

    if view.last_highlight is not None:
        row, col = divmod(view.last_highlight, 3)
        cell_rect = pygame.Rect(view.x + col * CELL_SIZE + 4, BOARD_Y + row * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8)
        color = PRUNED if view.prune_flash else HILITE
        pygame.draw.rect(SCREEN, color, cell_rect, width=4, border_radius=10)

    if view.final_move is not None and view.move_applied:
        row, col = divmod(view.final_move, 3)
        move_rect = pygame.Rect(view.x + col * CELL_SIZE + 10, BOARD_Y + row * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
        pygame.draw.rect(SCREEN, view.accent, move_rect, width=4, border_radius=12)

    for idx, mark in enumerate(view.board):
        if mark == "":
            continue
        row, col = divmod(idx, 3)
        cx = view.x + col * CELL_SIZE + CELL_SIZE // 2
        cy = BOARD_Y + row * CELL_SIZE + CELL_SIZE // 2
        if mark == "X":
            delta = 34
            pygame.draw.line(SCREEN, X_COLOR, (cx - delta, cy - delta), (cx + delta, cy + delta), 8)
            pygame.draw.line(SCREEN, X_COLOR, (cx + delta, cy - delta), (cx - delta, cy + delta), 8)
        else:
            pygame.draw.circle(SCREEN, O_COLOR, (cx, cy), 42, 8)

    if view.win_line is not None:
        a, _, c = view.win_line
        ax, ay = line_point_from_index(view.x, BOARD_Y, a)
        cx, cy = line_point_from_index(view.x, BOARD_Y, c)
        pygame.draw.line(SCREEN, WIN_LINE, (ax, ay), (cx, cy), 10)

    info_y = 570
    draw_text(f"Status: {view.status_text}", BODY_FONT, TEXT, view.x, info_y)
    nodes = view.result.nodes_visited if view.result else 0
    elapsed = view.result.elapsed_ms if view.result else 0.0
    score = view.result.score if view.result else "-"
    draw_text(f"Nodes visited: {nodes}", BODY_FONT, TEXT, view.x, info_y + 38)
    draw_text(f"Search time: {elapsed:.3f} ms", BODY_FONT, TEXT, view.x, info_y + 72)
    draw_text(f"Best score: {score}", BODY_FONT, TEXT, view.x, info_y + 106)

    if view.name == "Alpha-Beta" and view.result:
        prunes = sum(1 for s in view.result.steps if s.kind == "prune")
        draw_text(f"Pruning events: {prunes}", BODY_FONT, TEXT, view.x, info_y + 140)
    else:
        draw_text("Pruning events: N/A", BODY_FONT, MUTED, view.x, info_y + 140)

    draw_text("Yellow = exploring | Red = prune cutoff", SMALL_FONT, MUTED, view.x, info_y + 182)
    draw_text(f"Shared baseline board: {' '.join(cell or '-' for cell in baseline_board)}", SMALL_FONT, MUTED, view.x, info_y + 208)



def line_point_from_index(board_x: int, board_y: int, idx: int) -> Tuple[int, int]:
    row, col = divmod(idx, 3)
    return board_x + col * CELL_SIZE + CELL_SIZE // 2, board_y + row * CELL_SIZE + CELL_SIZE // 2


class ComparisonApp:
    def __init__(self) -> None:
        self.minimax_solver = MinimaxSolver()
        self.alphabeta_solver = AlphaBetaSolver()
        self.turn = 0
        self.global_board = ["", "", "", "", "", "", "", "", ""]
        self.current_player = "X"
        self.phase = "search"
        self.game_over = False
        self.end_timer = 0
        self.result_message = ""

        self.left_view = BoardViewState("Minimax", ACCENT_MINIMAX, LEFT_X, self.global_board[:], self.minimax_solver)
        self.right_view = BoardViewState("Alpha-Beta", ACCENT_AB, RIGHT_X, self.global_board[:], self.alphabeta_solver)
        self.start_new_round()

    def start_new_round(self) -> None:
        self.left_view.reset_for_search(self.global_board)
        self.right_view.reset_for_search(self.global_board)
        self.left_view.result = self.minimax_solver.choose_move(self.global_board[:], self.current_player)
        self.right_view.result = self.alphabeta_solver.choose_move(self.global_board[:], self.current_player)
        self.phase = "search"

    def reset(self) -> None:
        self.turn = 0
        self.global_board = ["", "", "", "", "", "", "", "", ""]
        self.current_player = "X"
        self.game_over = False
        self.end_timer = 0
        self.result_message = ""
        self.start_new_round()

    def update_view_animation(self, view: BoardViewState, dt_ms: int) -> None:
        if not view.result:
            return

        if view.searching:
            view.step_timer_ms += dt_ms
            if view.step_timer_ms >= SEARCH_HILITE_MS:
                view.step_timer_ms = 0
                if view.active_step_index < len(view.result.steps):
                    step = view.result.steps[view.active_step_index]
                    view.active_step_index += 1
                    view.last_highlight = step.move_index
                    view.prune_flash = step.kind == "prune"
                    if step.kind == "visit":
                        ab = f" | α={step.alpha} β={step.beta}" if step.alpha is not None and step.beta is not None else ""
                        view.status_text = f"Exploring move {step.move_index} at depth {step.depth}{ab}"
                    elif step.kind == "return":
                        view.last_return_value = step.value
                        ab = f" | α={step.alpha} β={step.beta}" if step.alpha is not None and step.beta is not None else ""
                        view.status_text = f"Returned score {step.value} from move {step.move_index}{ab}"
                    elif step.kind == "prune":
                        view.status_text = f"Pruned branch after move {step.move_index}"
                else:
                    view.searching = False
                    view.move_apply_timer_ms = 0
                    view.status_text = f"Best move found: {view.result.move}"
        elif not view.move_applied:
            view.move_apply_timer_ms += dt_ms
            if view.move_apply_timer_ms >= MOVE_APPLY_DELAY_MS:
                view.final_move = view.result.move
                view.board[view.final_move] = self.current_player
                winner, win_line = check_winner(view.board)
                view.winner = winner
                view.win_line = win_line
                view.move_applied = True
                if winner is None:
                    view.status_text = f"Applied move {view.final_move}"
                elif winner == "Draw":
                    view.status_text = "Draw"
                else:
                    view.status_text = f"Winner: {winner}"

    def update(self, dt_ms: int) -> None:
        if self.game_over:
            self.end_timer += dt_ms
            if self.end_timer >= END_DELAY_MS:
                self.reset()
            return

        self.update_view_animation(self.left_view, dt_ms)
        self.update_view_animation(self.right_view, dt_ms)

        if self.phase == "search":
            if self.left_view.move_applied and self.right_view.move_applied:
                # Keep the shared baseline consistent using the Alpha-Beta move.
                # Both algorithms should choose an optimal move; ties may differ,
                # so for synchronized progression we use the Alpha-Beta choice.
                chosen_move = self.right_view.final_move
                if chosen_move is None:
                    return
                self.global_board[chosen_move] = self.current_player
                winner, _ = check_winner(self.global_board)
                if winner is not None:
                    self.game_over = True
                    if winner == "Draw":
                        self.result_message = "Shared outcome: Draw"
                    else:
                        self.result_message = f"Shared outcome: {winner} wins"
                    return

                self.current_player = other_player(self.current_player)
                self.turn += 1
                self.start_new_round()

    def draw_header(self) -> None:
        draw_text("Minimax vs. Alpha-Beta Pruning", TITLE_FONT, TEXT, 40, 28)
        draw_text("Two synchronized tic-tac-toe searches, animated side by side", SUBTITLE_FONT, MUTED, 40, 66)
        draw_text("Alpha-Beta should reach the same best move while visiting fewer nodes and usually taking less time.", BODY_FONT, MUTED, 40, 98)

        help_text = "SPACE / R = restart   |   ESC = quit"
        draw_text(help_text, BODY_FONT, MUTED, WIDTH - 320, 38)

        speedup = 0.0
        if self.left_view.result and self.right_view.result and self.right_view.result.elapsed_ms > 0:
            speedup = self.left_view.result.elapsed_ms / self.right_view.result.elapsed_ms
        diff_nodes = 0
        if self.left_view.result and self.right_view.result:
            diff_nodes = self.left_view.result.nodes_visited - self.right_view.result.nodes_visited

        compare_rect = pygame.Rect(WIDTH // 2 - 240, 122, 480, 40)
        pygame.draw.rect(SCREEN, (38, 42, 56), compare_rect, border_radius=10)
        pygame.draw.rect(SCREEN, (90, 96, 120), compare_rect, width=1, border_radius=10)
        summary = f"Node savings: {diff_nodes:+d}    |    Time ratio (Minimax / Alpha-Beta): {speedup:.2f}x"
        draw_centered_text(summary, BODY_FONT, TEXT, compare_rect)

        if self.result_message:
            outcome_rect = pygame.Rect(WIDTH // 2 - 180, 700, 360, 38)
            pygame.draw.rect(SCREEN, (38, 56, 44), outcome_rect, border_radius=10)
            pygame.draw.rect(SCREEN, WIN_LINE, outcome_rect, width=2, border_radius=10)
            draw_centered_text(self.result_message, BODY_FONT, TEXT, outcome_rect)

    def draw(self) -> None:
        SCREEN.fill(BG)
        self.draw_header()
        draw_panel(self.left_view, self.current_player, self.turn, self.global_board)
        draw_panel(self.right_view, self.current_player, self.turn, self.global_board)
        pygame.display.flip()


def main() -> None:
    app = ComparisonApp()
    running = True

    while running:
        dt_ms = CLOCK.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_r):
                    app.reset()

        app.update(dt_ms)
        app.draw()

    pygame.quit()


if __name__ == "__main__":
    main()
