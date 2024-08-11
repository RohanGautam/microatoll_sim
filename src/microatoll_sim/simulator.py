# %%
import numpy as np
from numba import njit
from dataclasses import dataclass, field


@dataclass
class SimParams:
    """
    Contains the parameters used to control the coral growth simulation.
    - `vert_shift`: (units: m) Added to the provided sea level curve, to simulate the coral being present higher/lower on the reef
    - `gr_mean`: (units: mm/yr) The growth rate, how thick the coral grows per year
    - `d` : (units: mm) Controls how finely the new coral band should be resampled. Represents the arc-length distance between resampled points.
    - `dt`: (units: year) 1/dt is how many bands grow in a year. This controls the resolution of the bands. It can be made very fine for more precise simulation.
    - `T0`: (unit: year) When in the sea level time series to start the simulation
    - `delta_T`: (unit: years) Time period for coral growth simulation
    - `init_radius`: (unit: m) Initial radius of the coral, from which the simulation starts
    - `num_initial_points`: Sampling frequency for initial coral geometry
    """

    vert_shift: float  # m
    gr_mean: float  # mm/yr
    d: float  # mm
    dt: float  # year
    T0: int  # year
    delta_T: int  # years
    init_radius: float  # m
    num_initial_pts = 500
    NT: int = field(init=False)

    def __post_init__(self):
        self.delta_T *= 1.00001
        self.d = self.d / 1e3
        self.gr_mean = (self.gr_mean / 1e3) * self.dt
        self.NT = int(self.delta_T / self.dt)


@njit
def lowest_discreet(sl_df_arr, dx, Xmin, Xmax):
    Nx = round((Xmax - Xmin) / dx)
    Dout = np.zeros((Nx, 2))
    for i in range(Nx):
        Dout[i, 0] = Xmin + i * dx
        p0 = Dout[i, 0] - dx / 2
        p1 = Dout[i, 0] + dx / 2

        if p0 < sl_df_arr[0, 0]:
            Dout[i, 1] = sl_df_arr[0, 1]
            continue
        if p1 > sl_df_arr[-1, 0]:
            Dout[i, 1] = sl_df_arr[-1, 1]
            continue

        mn = np.inf
        for j in range(len(sl_df_arr)):
            if p0 < sl_df_arr[j, 0] < p1 and sl_df_arr[j, 1] < mn:
                mn = sl_df_arr[j, 1]

        if mn == np.inf:
            Ip = len(sl_df_arr)
            for j in range(len(sl_df_arr)):
                if Dout[i, 0] < sl_df_arr[j, 0]:
                    Ip = j
                    break
            a = (Dout[i, 0] - sl_df_arr[Ip - 1, 0]) / (
                sl_df_arr[Ip, 0] - sl_df_arr[Ip - 1, 0]
            )
            mn = sl_df_arr[Ip - 1, 1] * (1 - a) + sl_df_arr[Ip, 1] * a

        Dout[i, 1] = mn

    # return pd.DataFrame(Dout, columns=list(sl_df_arr.columns))
    return Dout


@njit
def arc(radius, Np):
    theta = np.linspace(0, np.pi / 2, Np)
    return radius * np.vstack((np.sin(theta), np.cos(theta))).T


@njit
def get_pointwise_unit_normals(line):
    """Get per-point normals for a line. Tangents estimated for a point using it's left and right neighbours. For extreme points, just one of it's neighbours are used. The tangents are then rotated to get the normals."""
    # finite differences but considering both neighbours of a point
    linelen = line.shape[0]
    normals = np.zeros_like(line)
    for i in range(linelen):
        if i == 0:
            tx, ty = line[1, 0] - line[0, 0], line[1, 1] - line[0, 1]
        elif i == linelen - 1:
            tx, ty = line[-1, 0] - line[-2, 0], line[-1, 1] - line[-2, 1]
        else:
            tx, ty = line[i + 1, 0] - line[i - 1, 0], line[i + 1, 1] - line[i - 1, 1]
        length = np.sqrt(tx**2 + ty**2)
        normals[i, 0] = -ty / length
        normals[i, 1] = tx / length
    return normals


@njit
def ccw(A, B, C):
    """Check if points A, B, C are listed in counterclockwise order"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


@njit
def check_cross(A, B, C, D):
    """Check if line segments AB and CD intersect"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


@njit
def kill_loop(line, living_status, gr):
    linelen = line.shape[0]
    for i in range(linelen - 1):
        if living_status[i] == 1:
            # Check for crossing with previous segments
            for j in range(i - 2, -1, -1):
                if check_cross(line[i, :], line[i + 1, :], line[j, :], line[j + 1, :]):
                    living_status[i:j:-1] = 0
                    break
            # Check for crossing with next segments
            for j in range(i + 2, linelen - 1):
                if check_cross(line[i, :], line[i + 1, :], line[j, :], line[j + 1, :]):
                    living_status[i:j] = 0
                    break

    normals = get_pointwise_unit_normals(line)
    line += normals * gr * living_status[:, np.newaxis]
    # Kill the parts where x coordinate is less than 0
    line[:, 0][line[:, 0] < 0] = 0
    living_status[line[:, 0] < 0] = 0

    return line, living_status


@njit
def resample(line, living_status, d):
    # Calculate cumulative sum of inter-point euclidean distances
    # deltas = np.diff(line, axis=0)
    deltas = line[1:, :] - line[:-1, :]
    distances = np.zeros(len(line))
    distances[1:] = np.sqrt(np.sum(deltas**2, axis=1))
    distances = np.cumsum(distances)

    # Calculate half distances for living status interpolation
    distances_half = np.zeros(len(line))
    distances_half[1:] = (distances[:-1] + distances[1:]) / 2

    # Number of points in the resampled line
    resampled_num_pts = int(np.floor(distances[-1] / d)) + 2
    resampled_line = np.zeros((resampled_num_pts, 2))
    resampled_line[0, :] = line[0, :]
    resampled_line[-1, :] = line[-1, :]
    new_living_status = np.zeros(resampled_num_pts)
    new_living_status[0] = living_status[0]
    new_living_status[-1] = living_status[-1]

    curr_vertex = 0
    living_status_vertex = 0
    for i in range(1, resampled_num_pts - 1):
        dsum = d * i

        while curr_vertex < len(distances) and dsum > distances[curr_vertex]:
            curr_vertex += 1

        a = (dsum - distances[curr_vertex - 1]) / (
            distances[curr_vertex] - distances[curr_vertex - 1]
        )
        p1, p2 = line[curr_vertex - 1, :], line[curr_vertex, :]
        resampled_line[i, 0] = p1[0] * (1 - a) + p2[0] * a
        resampled_line[i, 1] = p1[1] * (1 - a) + p2[1] * a

        # Update living status based on half distances
        while (
            living_status_vertex < len(distances_half)
            and dsum > distances_half[living_status_vertex]
        ):
            living_status_vertex += 1

        new_living_status[i] = living_status[living_status_vertex - 1]

    return resampled_line, new_living_status


@njit(cache=True)
def coral_growth_all(init_radius, num_initial_pts, d, gr_vec, NT, sl_arr):
    # Initialize line and living status
    init_line, init_living_status = resample(
        arc(init_radius, num_initial_pts), np.ones(num_initial_pts), d
    )
    lines = [init_line]
    living_statuses = [init_living_status]

    for it in range(NT):
        cur_line = lines[-1]
        cur_living_status = living_statuses[-1]
        # Update living status
        present_sea_level = sl_arr[it]
        cur_living_status[cur_line[:, 1] > present_sea_level] = 0
        # Grow
        new_line, new_living_status = kill_loop(cur_line, cur_living_status, gr_vec[it])

        # Ensure first x coord and last y coord are zero
        new_line[0, 0] = 0
        new_line[-1, 1] = 0
        # Resample
        new_line, new_living_status = resample(new_line, new_living_status, d)
        # Store history
        lines.append(new_line)
        living_statuses.append(new_living_status)

    return lines, living_statuses
