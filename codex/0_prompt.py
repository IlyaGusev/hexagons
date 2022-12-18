# Program painting on a hexagonal grid

COLORS = ["white", "black", "yellow", "green", "red", "blue", "purple", "orange"]

class HexagonalGrid:
    def __init__(self, rows_count: int = 10, cols_count: int = 18):
        # 0 column and 0 row are not used, indices are 1-based.
        self.grid = [[0 for _ in range(cols_count + 1)] for _ in range(rows_count + 1)]
        self.actions_log = []

    def paint_cell(self, row, col, color):
        assert color in COLORS, f"Wrong color: {color}"
        assert 1 <= row <= 10, f"Wrong row index: {row}"
        assert 1 <= col <= 18, f"Wrong col index: {col}"
        self.actions_log.append((row, col, color))
        self.grid[row][col] = COLORS.index(color)

    def get_actions(self):
        return sorted(self.actions_log)

    def clean_actions(self):
        self.actions_log = []

grid = HexagonalGrid(rows_count=10, cols_count=18)

# 
