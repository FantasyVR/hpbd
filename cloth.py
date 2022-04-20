import taichi as ti
import numpy as np

ti.init(arch=ti.cuda, dynamic_index=True)
N = 5
NV = (N + 1)**2
NT = 2 * N**2
NE = 2 * N * (N + 1) + N**2
pos = ti.Vector.field(3, ti.f32, shape=NV)
tri = ti.field(ti.i32, shape=3 * NT)
edge = ti.Vector.field(2, ti.i32, shape=NE)

old_pos = ti.Vector.field(3, ti.f32, NV)
inv_mass = ti.field(ti.f32, NV)
vel = ti.Vector.field(3, ti.f32, NV)
rest_len = ti.field(ti.f32, NE)
h = 0.01
MaxIte = 100

paused = ti.field(ti.i32, shape=())

# <<<<<<<<<<<<<<< For Hierarchical PBD >>>>>>>>>>>>>>>>>
# Particle restriction parameter
K = 2
# Each element store neighbor's indices, adj means adjacent
adj_vertices = ti.Vector.field(8, ti.i32, shape=NV)
# Store number of neighbors
adj_len = ti.field(ti.i32, shape=NV)
# Store whether a particle is coarse(-1) or fine(1)
is_fine = ti.field(ti.i32, shape=NV)
# each particle's color
per_vertex_color = ti.Vector.field(3, ti.f32, shape=NV)

# level 1 distance constraints particle indices
c_l1 = ti.Vector.field(2, ti.i32, shape=NE)
# level 1 distance constraints rest length
rest_len_l1 = ti.field(ti.f32, shape=NE)
# number of l1 constraints
num_l1_c = ti.field(ti.i32, shape=())


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        pos[idx] = ti.Vector([i / N, 0.5, j / N])
        inv_mass[idx] = 1.0
    inv_mass[N] = 0.0
    inv_mass[NV - 1] = 0.0


@ti.kernel
def init_tri():
    for i, j in ti.ndrange(N, N):
        tri_idx = 6 * (i * N + j)
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 2
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2
        else:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 1
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx + 1
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2


@ti.kernel
def init_edge():
    for i, j in ti.ndrange(N + 1, N):
        edge_idx = i * N + j
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        edge_idx = start + j * N + i
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
    start = 2 * N * (N + 1)
    for i, j in ti.ndrange(N, N):
        edge_idx = start + i * N + j
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
        else:
            edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])
    for i in range(NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        rest_len[i] = (p1 - p2).norm()


@ti.kernel
def init_neighbors():
    ti.loop_config(serialize=True)
    for i in range(NE):
        a, b = edge[i]
        l1, l2 = adj_len[a], adj_len[b]
        adj_vertices[a][l1] = b
        adj_vertices[b][l2] = a
        ti.atomic_add(adj_len[a], 1)
        ti.atomic_add(adj_len[b], 1)

    # max_adj_len = -1
    # ti.loop_config(serialize=True)
    # for i in range(NV):
    #     print(i, "-th particle's coarse neighbor: ", adj_len[i])
    #     if adj_len[i] > max_adj_len:
    #         max_adj_len = adj_len[i]
    # print("max adjacent neighbors: ", max_adj_len)


"""
compute number of coarse neighbors of particle p_idx
"""


@ti.func
def compute_num_coarse_neighbor(p_idx):
    neighbors = adj_vertices[p_idx]
    # print("neighbors indices: ", neighbors)
    num_neighbors = adj_len[p_idx]
    # print("num of neighbors of ", p_idx, ": ", num_neighbors)
    num_coarse_neighbor = 0
    for idx in range(num_neighbors):
        if is_fine[neighbors[idx]] < 0:  # coarse neighbor
            num_coarse_neighbor += 1
    return num_coarse_neighbor


"""
Test if one particle's all fine neighbors' coarse neighbors is larger than K.
:return  True( all fine neighbors' coarse neighbors>K), False(otherwise).
"""


@ti.func
def fine_neighbor_validation(p_idx):
    neighbors = adj_vertices[p_idx]
    num_neighbors = adj_len[p_idx]
    is_validate = 1
    for idx in range(num_neighbors):
        neighbor_idx = neighbors[idx]
        if is_fine[neighbor_idx] < 0:  # filter coarse neighbor
            continue
        # find the number of coarse neighbors of fine neighbor neighbor_idx
        num_coarse_neighbor = compute_num_coarse_neighbor(neighbor_idx)
        if num_coarse_neighbor <= K:
            is_validate = 0
    return is_validate


@ti.kernel
def particle_restriction():
    ti.loop_config(serialize=True)
    for i in range(NV):
        nce = compute_num_coarse_neighbor(i)
        # print(i, "-th particle's coarse neighbor: ", nce)
        if nce >= K and fine_neighbor_validation(i):
            is_fine[i] = 1


@ti.kernel
def init_particle_colors():
    for i in range(NV):
        if is_fine[i] > 0:  # fine particle
            per_vertex_color[i] = ti.Vector([1.0, 1.0, 1.0])
        else:  # coarse particle
            per_vertex_color[i] = ti.Vector([1.0, 0.0, 0.0])


# @ti.func
# def copy_from_l0_constraint():
#     for i in range(NE):
#         c_l1[i] = edge[i]


def find_p_j(p_i, pos_np, neighbors_np, num_adjs, is_fine_np):
    neighbors = neighbors_np[p_i]
    neighbors_len = num_adjs[p_i]
    cn = []
    for i in range(neighbors_len):
        if is_fine_np[neighbors[i]] < 0:
            cn.append(neighbors[i])
    num_coarse_neighbors = len(cn)
    avg_position = np.zeros(3, dtype=np.float32)
    for j in range(num_coarse_neighbors):
        avg_position += pos_np[cn[j]]
    avg_position /= num_coarse_neighbors

    nearest_neighbor = cn[0]
    min_dis = np.linalg.norm(pos_np[nearest_neighbor] - pos_np[p_i])
    for j in range(1, num_coarse_neighbors):
        distance = np.linalg.norm(pos_np[cn[j]] - pos_np[p_i])
        if distance < min_dis:
            nearest_neighbor = cn[j]
            min_dis = distance
    return nearest_neighbor


def remove_constraint(c_l1, c_l1_rl, p_i, p_j):
    for idx, c in enumerate(c_l1):
        c_d1, c_d2 = c
        if (c_d1 == p_i and c_d2 == p_j) or (c_d1 == p_j and c_d2 == p_i):
            return np.delete(c_l1, idx, 0), np.delete(c_l1_rl, idx, 0)
    return c_l1, c_l1_rl


def add_constraint(c_l1, c_l1_rl, p_k, p_j, pos_np):
    new_c_rest_len = np.linalg.norm(pos_np[p_k] - pos_np[p_j])
    new_c_l1 = np.vstack([c_l1, np.array([p_k, p_j])])
    new_c_l1_rl = np.append(c_l1_rl, new_c_rest_len)
    return new_c_l1, new_c_l1_rl


def constraint_restriction():
    pos_np = pos.to_numpy()
    c_l0_np = edge.to_numpy()
    c_l0_rl = rest_len.to_numpy()
    adj_matrix = adj_vertices.to_numpy()
    num_adjs = adj_len.to_numpy()
    is_fine_np = is_fine.to_numpy()

    c_l1 = c_l0_np
    c_l1_rl = c_l0_rl
    for p_i in range(NV):
        if is_fine_np[p_i] < 0:  # filter coarse particles
            continue
        p_j = find_p_j(p_i, pos_np, adj_matrix, num_adjs, is_fine_np)
        adj_particles = adj_matrix[p_i]
        np_len = num_adjs[p_i]
        for k in range(np_len):
            p_k = adj_particles[k]
            if p_k == p_j:
                # remove c(p_i, p_j)
                c_l1, c_l1_rl = remove_constraint(c_l1, c_l1_rl, p_i, p_j)
            elif p_k in adj_matrix[p_j]:
                # p_k is p_j's neighbor
                c_l1, c_l1_rl = remove_constraint(c_l1, c_l1_rl, p_i, p_k)
            else:  # p_k is not p_j's neighbor
                c_l1, c_l1_rl = add_constraint(c_l1, c_l1_rl, p_k, p_j, pos_np)


@ti.kernel
def semi_euler():
    gravity = ti.Vector([0.0, -0.1, 0.0])
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] += h * gravity
            old_pos[i] = pos[i]
            pos[i] += h * vel[i]


@ti.kernel
def solve_constraints():
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = pos[idx0] - pos[idx1]
        constraint = dis.norm() - rest_len[i]
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        if invM0 != 0.0:
            pos[idx0] += 0.5 * invM0 * l * gradient
        if invM1 != 0.0:
            pos[idx1] -= 0.5 * invM1 * l * gradient


@ti.kernel
def update_vel():
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / h


@ti.kernel
def collision():
    for i in range(NV):
        if pos[i][2] < -2.0:
            pos[i][2] = 0.0


def step():
    semi_euler()
    for i in range(MaxIte):
        solve_constraints()
        collision()
    update_vel()


def init():
    init_pos()
    init_tri()
    init_edge()
    # For Hierarchical PBD
    init_neighbors()
    is_fine.fill(-1)  # init all particles as coarse
    particle_restriction()
    init_particle_colors()
    constraint_restriction()


init()
window = ti.ui.Window("Display Mesh", (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 0.0, 2.5)
camera.lookat(0.5, 0.5, 0.0)
camera.fov(90)

paused[None] = 1
while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE]:
            exit()
    if window.is_pressed(ti.ui.SPACE):
        paused[None] = not paused[None]

    if not paused[None]:
        step()
        paused[None] = not paused[None]

    camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

    scene.mesh(pos, tri, color=(0.0, 1.0, 0.0), two_sided=True)
    scene.particles(pos, radius=0.04, per_vertex_color=per_vertex_color)
    canvas.scene(scene)
    window.show()