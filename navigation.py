"""Occupancy grid flattening, 8-connected A* search, and spatial vectors."""
import heapq
import numpy as np
import config

class Navigator:
    def __init__(self):
        self.res = config.GRID_RESOLUTION

    def pointcloud_to_grid(self, pcd, grid_span=10.0):
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return np.zeros((100, 100), dtype=np.uint8)

        grid_cells = int(grid_span / self.res)
        grid = np.zeros((grid_cells, grid_cells), dtype=np.uint8)
        
        # Flatten onto X-Z plane
        for p in points:
            gx = int((p[0] + grid_span / 2) / self.res)
            gz = int((p[2] + grid_span / 2) / self.res)
            if 0 <= gx < grid_cells and 0 <= gz < grid_cells:
                grid[gx, gz] = 1
        return grid

    def a_star(self, grid, start, goal):
        rows, cols = grid.shape
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        def h(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
            for dx, dz in neighbors:
                nbr = (current[0] + dx, current[1] + dz)
                if 0 <= nbr[0] < rows and 0 <= nbr[1] < cols:
                    if grid[nbr[0], nbr[1]] == 1:
                        continue  # Obstacle
                    tentative_g = g_score[current] + np.hypot(dx, dz)
                    if tentative_g < g_score.get(nbr, float("inf")):
                        came_from[nbr] = current
                        g_score[nbr] = tentative_g
                        heapq.heappush(open_set, (tentative_g + h(nbr, goal), nbr))
        return None

    def get_spatial_direction(self, current_pos, target_pos):
        dx = target_pos[0] - current_pos[0]
        dz = target_pos[2] - current_pos[2]
        dist = np.sqrt(dx**2 + dz**2)
        angle = np.degrees(np.arctan2(dx, dz))

        if abs(angle) < 15:
            direction = "straight ahead"
        elif angle >= 15:
            direction = f"turn right by {int(angle)} degrees"
        else:
            direction = f"turn left by {int(abs(angle))} degrees"

        return dist, direction
