import numpy as np
import matplotlib.pyplot as plt
from numba import njit

pi = np.pi


def plot_arc_segment(Ox, Oy, R, theta1, theta2):
    if theta1 != theta2:
        # Generate angles between theta1 and theta2
        if theta1 < theta2:
            theta = np.linspace(theta1, theta2, 100)
        else:
            theta = np.linspace(theta1, theta2 + 2 * np.pi, 100)

        # Calculate x and y coordinates of the arc
        x = Ox + R * np.cos(theta)
        y = Oy + R * np.sin(theta)

        # Plot the arc segment
        plt.plot(x, y)

        ## Plot the center
        # plt.plot(Ox, Oy, 'ro')

        # Set axis equal to maintain aspect ratio
        plt.axis("equal")

        # Label the plot
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Analytical approximation")


@njit
def EndPoint(Arcs, PT, GEN):
    # Get Coordinate of Endpoints
    # GEN: generation
    # PT:ID of arcsegment (0:BF, 1:MF, 2:TF, 3:TB)
    P = np.zeros((2, 2))
    P[0, 0] = Arcs[PT, GEN, 0] + Arcs[PT, GEN, 2] * np.cos(Arcs[PT, GEN, 3])
    P[0, 1] = Arcs[PT, GEN, 1] + Arcs[PT, GEN, 2] * np.sin(Arcs[PT, GEN, 3])
    P[1, 0] = Arcs[PT, GEN, 0] + Arcs[PT, GEN, 2] * np.cos(Arcs[PT, GEN, 4])
    P[1, 1] = Arcs[PT, GEN, 1] + Arcs[PT, GEN, 2] * np.sin(Arcs[PT, GEN, 4])
    return P


@njit
def FindC(P1, P2, V):
    # Get Parameters of MF
    M = np.zeros(2)  # Midpoint of the endpoints
    M[0] = (P1[0] + P2[0]) / 2
    M[1] = (P1[1] + P2[1]) / 2
    C = -(P2[0] - P1[0]) / (P2[1] - P1[1])
    O = np.zeros(2)
    O[0] = (-V * P1[0] + P1[1] + C * M[0] - M[1]) / (C - V)
    O[1] = (P1[0] - P1[1] / V - M[0] + M[1] / C) / (1 / C - 1 / V)
    dx = P1[0] - O[0]
    dy = P1[1] - O[1]
    R = np.sqrt(dx * dx + dy * dy)
    return O, R


@njit
def growth_Arc(gr, gen_current, sfc_current, SL, Arcs, DD, Sfc):
    I = gen_current
    Isf = sfc_current
    V = gr * (SL[I, 0] - SL[I - 1, 0])  # growth amount
    VTs = gr * SL[I, 0]  # Total growth amount

    C = np.zeros(2)  # 2nd endpoint of BF
    if DD[I - 1, 3] == 0:  # The last diedown was on BF
        C[0] = DD[I - 1, 0] + V * np.cos(DD[I - 1, 2])
        C[1] = DD[I - 1, 1] + V * np.sin(DD[I - 1, 2])
    else:
        P = EndPoint(Arcs, 0, I - 1)
        C[0] = P[1, 0] + V * np.cos(Arcs[0, I - 1, 4])
        C[1] = P[1, 1] + V * np.sin(Arcs[0, I - 1, 4])
    Ox = (C[0] * C[0] + C[1] * C[1] - VTs * VTs) / (2 * (C[0] - VTs))

    # Arc_BF
    Arcs[0, I, 0] = Ox
    Arcs[0, I, 1] = 0
    Arcs[0, I, 2] = VTs - Ox
    Arcs[0, I, 3] = 0
    Arcs[0, I, 4] = np.arctan2(C[1], C[0] - Ox)

    # Arc_TF
    Arcs[2, I, 0] = DD[I - 1, 0]
    Arcs[2, I, 1] = DD[I - 1, 1]
    Arcs[2, I, 2] = V
    Arcs[2, I, 3] = DD[I - 1, 2]
    Arcs[2, I, 4] = pi / 2

    # Arc_MF
    P = EndPoint(Arcs, 0, I)
    P1 = np.array([P[1, 0], P[1, 1]])
    P = EndPoint(Arcs, 2, I)
    P2 = np.array([P[0, 0], P[0, 1]])
    if P1[0] == P2[0]:
        Arcs[1, I, :] = Arcs[2, I, :]
        Arcs[1, I, 4] = Arcs[1, I, 3]
    else:
        O, R = FindC(P1, P2, C[1] / (C[0] - Ox))
        Arcs[1, I, 0] = O[0]
        Arcs[1, I, 1] = O[1]
        Arcs[1, I, 2] = R
        Arcs[1, I, 3] = Arcs[0, I - 1, 4]
        Arcs[1, I, 4] = np.arctan2(P2[1] - O[1], P2[0] - O[0])

    # Groove
    Vp = V
    temp = 0
    Grv = np.zeros(2)
    for j in range(Isf, -1, -1):
        L = Sfc[j, 2] * (Sfc[j, 4] - Sfc[j, 3])
        if L > Vp:
            temp = 1
            Jsf = j
            break
        else:
            Vp = Vp - L

    if temp == 1:
        phi1 = Vp / Sfc[Jsf, 2]
        phi2 = Sfc[Jsf, 3] + phi1
        Grv[0] = Sfc[Jsf, 0] + Sfc[Jsf, 2] * np.cos(phi2)
        Grv[1] = Sfc[Jsf, 1] + Sfc[Jsf, 2] * np.sin(phi2)

        # plt.plot(Grv[0], Grv[1], "bo")

        # Arc_TB
        Pk = np.zeros(2)
        Pk[0] = Arcs[2, I, 0]
        Pk[1] = Arcs[2, I, 1] + Arcs[2, I, 2]
        Oz = (
            Grv[0] * Grv[0]
            + Grv[1] * Grv[1]
            + Pk[0] * Pk[0]
            - Pk[1] * Pk[1]
            - 2 * Pk[0] * Grv[0]
        ) / (2 * (Grv[1] - Pk[1]))
        Arcs[3, I, 0] = Pk[0]
        Arcs[3, I, 1] = Oz
        Arcs[3, I, 2] = Pk[1] - Oz
        Arcs[3, I, 3] = pi / 2
        if Grv[1] > Oz:
            Arcs[3, I, 4] = np.arctan2(Grv[1] - Oz, Grv[0] - Pk[0])
        else:
            Arcs[3, I, 4] = 2 * pi + np.arctan2(Grv[1] - Oz, Grv[0] - Pk[0])
    else:
        Jsf = 0
        Arcs[3, I, 0] = DD[I - 1, 0]
        Arcs[3, I, 1] = DD[I - 1, 1]
        Arcs[3, I, 2] = V
        Arcs[3, I, 3] = pi / 2
        Arcs[3, I, 4] = pi / 2 + np.arcsin(DD[I - 1, 0] / V)
        Grv[0] = 0
        Grv[1] = DD[I - 1, 1] + V * np.cos(Arcs[3, I, 4])

    # Locate diedown point
    DD[I, 1] = SL[I, 1]
    P = EndPoint(Arcs, 1, I)
    if DD[I, 1] < C[1]:
        DD[I, 3] = 0
    elif DD[I, 1] < P[1, 1]:
        DD[I, 3] = 1
    else:
        DD[I, 3] = 2

    dz = DD[I, 1] - Arcs[int(DD[I, 3]), I, 1]
    dx = np.sqrt(Arcs[int(DD[I, 3]), I, 2] * Arcs[int(DD[I, 3]), I, 2] - dz * dz)
    DD[I, 0] = Arcs[int(DD[I, 3]), I, 0] + dx
    DD[I, 2] = np.arctan2(dz, dx)

    # Renew surface trace
    Isf = Jsf
    if temp == 0:
        Sfc[Isf, :] = Arcs[3, I, :]
    else:
        Sfc[Isf, 3] = phi2
        Isf = Isf + 1
        Sfc[Isf, :] = Arcs[3, I, :]

    for j in range(2, -1, -1):
        Isf = Isf + 1
        Sfc[Isf, :] = Arcs[j, I, :]
        if j == DD[I, 3]:
            break
    Sfc[Isf, 3] = DD[I, 2]

    return Isf, Arcs, DD, Sfc


@njit(error_model="numpy")
def growth(Nc, gr, Isf, SLin, Arcs, DD, Sfc):
    for i in range(1, Nc):
        Isf, Arcs, DD, Sfc = growth_Arc(gr, i, Isf, SLin, Arcs, DD, Sfc)
    return Isf, Arcs, DD, Sfc


# for j in range(0, Nc):
#     for i in range(0, 4):
#         plot_arc_segment(*Arcs[i, j, :])
#     plt.plot(DD[j, 0], DD[j, 1], "ro")

# plt.show()
