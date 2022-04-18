import numpy as np


class Camera:

    def __init__(self, position, up, yaw, pitch):
        self.position = position
        self.up = up
        self.right = np.zeros((3, 1), dtype=np.float32)
        self.front = self.right
        self.yaw = yaw
        self.pitch = pitch
        self.updateCameraVectors()
        self.view_matrix = np.identity(4, dtype=np.float32)
        self.projection_matrix = np.zeros((4, 4), dtype=np.float32)

    def updateCameraVectors(self):
        radians = lambda x: x * np.pi / 180
        yaw, pitch = radians(self.yaw), radians(self.pitch)
        x = np.cos(yaw) * np.cos(pitch)
        y = np.sin(pitch)
        z = np.sin(yaw) * np.cos(pitch)
        self.front = np.array([x, y, z])
        self.right = np.cross(self.front, self.up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up /= np.linalg.norm(self.up)

    def look_at(self, eye, center, up):
        f = (center - eye) / np.linalg.norm(center - eye)
        s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
        u = np.cross(s, f)
        self.view_matrix[0, :-1] = s
        self.view_matrix[1, :-1] = u
        self.view_matrix[2, :-1] = -f
        self.view_matrix[0, 3] = -np.dot(s, eye)
        self.view_matrix[1, 3] = -np.dot(u, eye)
        self.view_matrix[2, 3] = np.dot(f, eye)
        return self.view_matrix

    def move(self, direction, speed):
        if direction == "w":
            self.position += self.front * speed
        elif direction == "s":
            self.position -= self.front * speed
        elif direction == "a":
            self.position -= self.right * speed
        elif direction == "d":
            self.position += self.right * speed

    def get_view(self):
        return self.look_at(self.position, self.position + self.front, self.up)

    def perspective(self, fov, aspect, near, far):
        tanHalfFovy = np.tan(fov / 2)
        self.projection_matrix[0, 0] = 1 / (aspect * tanHalfFovy)
        self.projection_matrix[1, 1] = 1 / (tanHalfFovy)
        self.projection_matrix[2, 2] = -(far + near) / (far - near)
        self.projection_matrix[3, 2] = -1.0
        self.projection_matrix[2, 3] = -2 * far * near / (far - near)
        return self.projection_matrix


def transform_positions(pos, affine_trans):
    for i in range(pos.shape[0]):
        ap = np.array([pos[i][0], pos[i][1], pos[i][2], 1.0])
        tap = affine_trans @ ap
        pos[i] = np.array([tap[0], tap[1], tap[2]]) / tap[3] * 0.5 + 0.5
    return pos


if __name__ == "__main__":
    eye = np.array([0, 0, -2], dtype=np.float32)
    center = np.array([0, 0, 0], dtype=np.float32)
    camera = Camera(eye, center, 90, 0)
    v = camera.get_view()
    p = camera.perspective(fov=np.pi / 4,
                           aspect=4.0 / 3.0,
                           near=0.1,
                           far=100.0)
    mvp = p @ v
