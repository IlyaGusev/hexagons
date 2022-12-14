import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

COLORS_MAPPING = {
    0: "white",
    1: "black",
    2: "yellow",
    3: "green",
    4: "red",
    5: "blue",
    6: "purple",
    7: "orange"
}


def draw_board(colors):
    radius = 2.0 / 3.0
    vertical_step = radius * np.sqrt(3)
    colors = [COLORS_MAPPING[c] for c in colors]

    # Horizontal cartesian coords
    hcoord = []
    for i in range(10):
        for j in range(18):
            hcoord.append(j)

    # Vertical cartersian coords
    vcoord = []
    for i in range(10):
        for j in range(18):
            vcoord.append(-i * vertical_step - (vertical_step / 2 if j % 2 == 1 else 0.0))

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_aspect('equal')

    # Add some coloured hexagons
    for x, y, c in zip(hcoord, vcoord, colors):
        hex = RegularPolygon(
            (x, y), numVertices=6, radius=radius, 
            orientation=np.radians(90),
            alpha=0.8,
            edgecolor='k',
            facecolor=c
        )
        ax.add_patch(hex)
    ax.scatter(hcoord, vcoord, alpha=0.0)
    plt.show()
