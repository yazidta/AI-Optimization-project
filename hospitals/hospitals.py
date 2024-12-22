import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

class Space():

    def __init__(self, height, width, num_hospitals):
        self.height = height
        self.width = width
        self.num_hospitals = num_hospitals
        self.houses = set()
        self.hospitals = set()

    def add_house(self, row, col):
        self.houses.add((row, col))

    def available_spaces(self):
        candidates = set(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        )
        for house in self.houses:
            candidates.remove(house)
        for hospital in self.hospitals:
            candidates.remove(hospital)
        return candidates

    def hill_climb_with_annealing(self, maximum=None, image_prefix=None):
        count = 0
        self.hospitals = set()
        trajectory = []
        visited_states = set()

        for i in range(self.num_hospitals):
            self.hospitals.add(random.choice(list(self.available_spaces())))
        trajectory.append(tuple(self.hospitals))
        visited_states.add(tuple(self.hospitals))

        temperature = 100  # Initial temperature
        cooling_rate = 0.99  # Decrease temperature

        if image_prefix:
            self.output_image(f"{image_prefix}{str(count).zfill(3)}.png")

        while maximum is None or count < maximum:
            count += 1
            best_neighbors = []
            best_neighbor_cost = None

            for hospital in self.hospitals:
                for replacement in self.get_neighbors(*hospital):
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    neighbor_tuple = tuple(neighbor)
                    if neighbor_tuple in visited_states:
                        continue

                    cost = self.get_cost(neighbor)
                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            current_cost = self.get_cost(self.hospitals)
            if best_neighbor_cost is None or best_neighbor_cost >= current_cost:
                # Accept worse solution based on probability
                if best_neighbor_cost is not None:
                    delta = best_neighbor_cost - current_cost
                    if random.random() > math.exp(-delta / temperature):
                        return self.hospitals, trajectory
                return self.hospitals, trajectory

            self.hospitals = random.choice(best_neighbors)
            trajectory.append(tuple(self.hospitals))
            visited_states.add(tuple(self.hospitals))

            if image_prefix:
                self.output_image(f"{image_prefix}{str(count).zfill(3)}.png")

            temperature *= cooling_rate  # Decrease temperature

    def get_cost(self, hospitals):
        cost = 0
        for house in self.houses:
            cost += min(
                abs(house[0] - hospital[0]) + abs(house[1] - hospital[1])
                for hospital in hospitals
            )
        return cost

    def get_neighbors(self, row, col):
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]
        neighbors = []
        for r, c in candidates:
            if (r, c) in self.houses or (r, c) in self.hospitals:
                continue
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbors.append((r, c))
        return neighbors

    def generate_heat_map(self):
        heat_map = np.zeros((self.height, self.width))

        for row in range(self.height):
            for col in range(self.width):
                heat_map[row, col] = sum(
                    abs(row - house[0]) + abs(col - house[1])
                    for house in self.houses
                )
        return heat_map

    def plot_heat_map(self, heat_map, trajectory):
        fig, ax = plt.subplots()
        c = ax.imshow(heat_map, cmap='hot', interpolation='nearest')
        plt.colorbar(c, ax=ax)

        # Flatten the trajectory of hospital positions
        all_coords = [coord for state in trajectory for coord in state]
        x, y = zip(*all_coords)
        
        ax.plot(y, x, marker='o', color='cyan', label='Optimization Path')
        ax.legend()
        plt.title("Heat Map with Optimization Path")
        plt.show()


    def output_image(self, filename):
        cell_size = 100
        cell_border = 2
        cost_size = 40
        padding = 10

        img = Image.new(
            "RGBA",
            (self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "white"
        )
        house = Image.open("assets/images/House.png").resize(
            (cell_size, cell_size)
        )
        hospital = Image.open("assets/images/Hospital.png").resize(
            (cell_size, cell_size)
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 30)
        draw = ImageDraw.Draw(img)

        for i in range(self.height):
            for j in range(self.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                draw.rectangle(rect, fill="black")

                if (i, j) in self.houses:
                    img.paste(house, rect[0], house)
                if (i, j) in self.hospitals:
                    img.paste(hospital, rect[0], hospital)

        draw.rectangle(
            (0, self.height * cell_size, self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "black"
        )
        draw.text(
            (padding, self.height * cell_size + padding),
            f"Cost: {self.get_cost(self.hospitals)}",
            fill="white",
            font=font
        )

        img.save(filename)

# Create a new space and add houses randomly
s = Space(height=10, width=20, num_hospitals=1)
for i in range(15):
    s.add_house(random.randrange(s.height), random.randrange(s.width))

# Generate heat map
heat_map = s.generate_heat_map()

# Perform hill climbing with simulated annealing
hospitals, trajectory = s.hill_climb_with_annealing(image_prefix="hospitals")

# Plot heat map with optimization path
s.plot_heat_map(heat_map, trajectory)
