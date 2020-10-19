
import numpy as np
import cv2

"""construct single agent lidar observation
"""
def construct_single_lidar_obs(env_obs, args):
    # cv2.imwrite('ogm.png', env_obs.occupancy_grid_map.data)
    # cv2.imwrite('rgb.png', env_obs.top_down_rgb.data)
    # cv2.imwrite('dagm.png', env_obs.drivable_area_grid_map.data)
    #print(env_obs.neighborhood_vehicle_states)
    # agent characteristics
    ego_state = env_obs.ego_vehicle_state

    ego_position         = ego_state.position 
    ego_bounding_box     = np.array([ego_state.bounding_box.length,
                                     ego_state.bounding_box.width,
                                    ego_state.bounding_box.height])
    ego_heading          = np.array(ego_state.heading)
    ego_steering         = np.array(ego_state.steering)
    ego_linear_velocity  = ego_state.linear_velocity
    ego_angular_velocity = ego_state.angular_velocity
    ego_line             = np.array(ego_state.lane_index)

    # lidar info
    lidar_ratio = args.construct_obs.lidar_ratio





