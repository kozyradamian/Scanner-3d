# Map Depth Frame to World Space
# returns the DepthFrame mapped to camera space
def depth_2_world(kinect, depth_frame_data, camera_space_point, as_array=False):
    import numpy as np
    import ctypes
    depth2world_points_type = camera_space_point * np.int(512 * 424)
    depth2world_points = ctypes.cast(depth2world_points_type(), ctypes.POINTER(camera_space_point))
    kinect._mapper.MapDepthFrameToCameraSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(512 * 424), depth2world_points)
    points = ctypes.cast(depth2world_points, ctypes.POINTER(ctypes.c_float))
    data = np.ctypeslib.as_array(points, shape=(424, 512, 3))
    if not as_array:
        return depth2world_points
    else:
        return data
