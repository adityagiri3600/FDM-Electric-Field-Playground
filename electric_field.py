import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


"""
simulation of the electric wave

for simulation purposes only the header variables, set_permittivity and source are important

this program works on the principle of FDM and uses mpl.animation to animate

"""


N = 500  # number of divisions
margin = 100
dt = 0.005
max_time = 5
fps = 100
color_map = "viridis"
# dt/ε ≤ 0.005 for stability. ε is the minimum permittivity


def set_permittivity(x, y):
    if inside_lens(x, y, 0.3, 0.5, radius=0.3, width=0.1, type="concave"):
        return 10
    else:
        return 1


def source(E, time):
    point_source(E, 0.1, 0.5)
    points_source(E, bresenham_line(0.4, 0.5, 0.7, 0.6))
    return E


# laplacian operator stencil
laplacian = (-2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)) / 0.01**2


# user uses coordinates from 0 to 1 while pixels range from margin to N-margin
def pixel(coordinate):
    return int(coordinate * (N - 2 * margin) + margin)


def coordinate(pixel):
    return (pixel - margin) / (N - 2 * margin)


def bresenham_line(x0, y0, x1, y1):
    # Bresenham's Line Algorithm
    # gives points on the line btn two points
    x0, y0, x1, y1 = pixel(x0), pixel(y0), pixel(x1), pixel(y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy
    line_points = []
    while True:
        line_points.append((coordinate(x0), coordinate(y0)))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return np.array(line_points)


def point_source(E, x, y, wavelength=0.1, A=1):
    E[pixel(x), pixel(y)] = A * np.sin(np.pi * time / (wavelength)) ** 2


def points_source(E, points, wavelength=0.1, A=1):
    points = np.vectorize(pixel)(points)
    E[points[:, 0], points[:, 1]] = A * np.sin(np.pi * time / (wavelength)) ** 2


def inside_lens(x, y, x0, y0, radius=0.3, aperture=0.4, width=0.1, type="convex"):
    """
    (x0,y0) is the location of the lens
    radius is radius of curvature
    aperture is the height of the lens
    for convex, aperture must be less than twice
    the radius, for concave, width is used instead
    of aperture
    """
    if type == "concave":
        return not (
            (x - x0 + width / 2 + radius) ** 2 + (y - y0) ** 2 < radius**2
            or (x - x0 - width / 2 - radius) ** 2 + (y - y0) ** 2 < radius**2
            or y > y0 + 0.75 * radius
            or y < y0 - 0.75 * radius
            or (x - x0) ** 2 + (y - y0) ** 2 > radius**2
        )
    else:
        if aperture > 2 * radius:
            raise Exception("aperture must be less than twice the radius")
        return (x - x0 + np.sqrt(radius**2 - (aperture**2) / 4)) ** 2 + (
            y - y0
        ) ** 2 < radius**2 and (
            x - x0 - np.sqrt(radius**2 - (aperture**2) / 4)
        ) ** 2 + (
            y - y0
        ) ** 2 < radius**2


permittivity = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        permittivity[i, j] = set_permittivity(coordinate(i), coordinate(j))
E = np.zeros((N, N))
E_old = E
time = 0


def update(E, E_old, time):
    # d²E/dt² = -∇²E/ε, wave equation
    E_new = (
        2 * E - E_old + (dt**2) * (E.dot(laplacian) + laplacian.dot(E)) / permittivity
    )
    E_new = source(E_new, time)
    return E_new


# plotting
fig, ax = plt.subplots()
ims = []

for i in range(int(max_time * fps)):
    E_old, E = E, update(E, E_old, time)
    time += dt
    im = plt.imshow(
        np.rot90(
            E[margin : N - margin, margin : N - margin]
            + 0.01 * permittivity[margin : N - margin, margin : N - margin]
        ),
        cmap=color_map,
        vmin=0,
        vmax=1,
        animated=True,
    )
    ims.append([im])
x_tick_positions = np.linspace(0, N - 2 * margin, 11)
x_tick_labels = "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1".split()
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_tick_labels)
y_tick_positions = np.linspace(N - 2 * margin, 0, 11)
y_tick_labels = "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1".split()
ax.set_yticks(y_tick_positions)
ax.set_yticklabels(y_tick_labels)
fig.tight_layout()
ani = animation.ArtistAnimation(fig, ims, interval=1000 / fps, blit=True)
writergif = animation.PillowWriter(fps=30)
ani.save(r"animation.gif", writer=writergif)
plt.show()

