import numpy as np
from KalmanFilter import kalman_filter

if __name__ == '__main__':
    position = np.array([[1, 2, 1], [2, 2, 2], [3, 3, 3]])
    velocity = np.array([[0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9]])

    #for n in range(200):
        #position = np.concatenate((position, np.array([[n, n + 1, n + 2]])))
        #velocity = np.concatenate((velocity, np.array([[n, n, n]])))

    actual_pos_x = []
    actual_pos_y = []
    actual_pos_z = []

    x_offset = 4
    y_offset = 2
    z_offset = 1

    while True:
        position = np.concatenate((position, np.array([[x_offset, y_offset, z_offset]])))
        velocity = np.concatenate((velocity, np.array([[0.5, 0, 0]])))

        actual_pos_x.append(kalman_filter(wsp='x', position=position[len(position)-1][0], velocity=velocity[len(position)-1][0]))
        actual_pos_y.append(kalman_filter(wsp='y', position=position[len(position)-1][1], velocity=velocity[len(velocity)-1][1]))
        actual_pos_z.append(kalman_filter(wsp='z', position=position[len(position)-1][2], velocity=velocity[len(velocity)-1][2]))

        x_offset += 0.02
        y_offset += 0.01
        z_offset += 0.01
        if x_offset > 100:
            break
    print(len(actual_pos_x))
    print(actual_pos_x[4790:])
    print(actual_pos_y[4790:])
    print(actual_pos_z[4790:])
