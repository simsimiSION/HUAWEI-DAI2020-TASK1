import numpy as np
from matplotlib.path import Path
from shapely.geometry import LineString
import math
import gym

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import OGM, NeighborhoodVehicles
from smarts.core.controllers import ActionSpaceType


"""
Coordinate：

      ^ y
      |
      |       x
-------------->
      |
      |
"""
import yaml

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

def get_args():
    """返回参数
    """
    f = open('/home/HUAWEI-DAI2020/task1/starter_kit/train_example/utils/params.yaml', 'r')
    file_data = f.read()
    f.close()

    return dict2obj(yaml.load(file_data, Loader=yaml.FullLoader))
# ====================================================================================
#    construction normal characteristic
# ====================================================================================

"""calculate ego vehicle relative position 
"""
def get_ego_position(ego_position,  
                     ego_heading, 
                     ego_bounding_box,
                     args,  
                     type='corner'):  # type: center/corner/threaten
    # center  
    if type == 'center':
        return ego_position


    ego_heading_cosine = np.cos(-ego_heading)
    ego_heading_sine = np.sin(-ego_heading)
    # threaten
    if type == 'threaten':
        l, h = ego_bounding_box
        lu = l / 2 + args.safety.threaten_distance_front
        lb = l / 2 + args.safety.threaten_distance_back
        h += args.safety.threaten_distance_left * 2

        ego_left_up_corner = ego_position + np.array([
            -h / 2 * ego_heading_cosine + lu * ego_heading_sine,
            lu * ego_heading_cosine - -h / 2 * ego_heading_sine]).reshape(-1)
        ego_right_up_corner = ego_position + np.array([
            h / 2 * ego_heading_cosine + lu * ego_heading_sine,
            lu * ego_heading_cosine - h / 2 * ego_heading_sine]).reshape(-1)
        ego_left_down_corner = ego_position + np.array([
            -h / 2 * ego_heading_cosine + -lb * ego_heading_sine,
            -lb * ego_heading_cosine - -h / 2 * ego_heading_sine]).reshape(-1)
        ego_right_down_corner = ego_position + np.array([
            h / 2 * ego_heading_cosine + -lb * ego_heading_sine,
            -lb * ego_heading_cosine - +h / 2 * ego_heading_sine]).reshape(-1)

        return np.array([ego_left_up_corner,
                         ego_right_up_corner,
                         ego_left_down_corner,
                         ego_right_down_corner])

    # corner
    if type == 'corner':
        l, h = ego_bounding_box

        ego_left_up_corner = ego_position + np.array([
            -h / 2 * ego_heading_cosine + l / 2 * ego_heading_sine,
            l / 2 * ego_heading_cosine - -h / 2 * ego_heading_sine]).reshape(-1)
        ego_right_up_corner = ego_position + np.array([
            h / 2 * ego_heading_cosine + l / 2 * ego_heading_sine,
            l / 2 * ego_heading_cosine - h / 2 * ego_heading_sine]).reshape(-1)
        ego_left_down_corner = ego_position + np.array([
            -h / 2 * ego_heading_cosine + -l / 2 * ego_heading_sine,
            -l / 2 * ego_heading_cosine - -h / 2 * ego_heading_sine]).reshape(-1)
        ego_right_down_corner = ego_position + np.array([
            h / 2 * ego_heading_cosine + -l / 2 * ego_heading_sine,
            -l / 2 * ego_heading_cosine - +h / 2 * ego_heading_sine]).reshape(-1)

        return np.array([ego_left_up_corner,
                         ego_right_up_corner,
                         ego_left_down_corner,
                         ego_right_down_corner])

    return None

"""calculate neighborhood vehicle relative position 
"""
def get_neighborhood_vehicle_relative_position(ego_position,  
                                               neighborhood_vehicle_position, 
                                               neighborhood_vehicle_bounding_box,  
                                               neighborhood_vehicle_heading,  
                                               type='corner'):  # type: center/corner

    relative_position = neighborhood_vehicle_position - ego_position

    # center  
    if type == 'center':
        return relative_position

    l, h = neighborhood_vehicle_bounding_box

    neighborhood_vehicle_heading_cosine = np.cos(-neighborhood_vehicle_heading)
    neighborhood_vehicle_heading_sine = np.sin(-neighborhood_vehicle_heading)

    # calculate 4 corner position
    # formula: 
    #   x_1 = x * cos(ø) + y * sin(ø)
    #   y_1 = y * cos(ø) - x * sin(ø)
    neighborhood_vehicle_left_up_corner = relative_position + np.array([
        -h / 2 * neighborhood_vehicle_heading_cosine + l / 2 * neighborhood_vehicle_heading_sine,
        l / 2 * neighborhood_vehicle_heading_cosine - -h / 2 * neighborhood_vehicle_heading_sine]).reshape(-1)
    neighborhood_vehicle_right_up_corner = relative_position + np.array([
        h / 2 * neighborhood_vehicle_heading_cosine + l / 2 * neighborhood_vehicle_heading_sine,
        l / 2 * neighborhood_vehicle_heading_cosine - h / 2 * neighborhood_vehicle_heading_sine]).reshape(-1)
    neighborhood_vehicle_left_down_corner = relative_position + np.array([
        -h / 2 * neighborhood_vehicle_heading_cosine + -l / 2 * neighborhood_vehicle_heading_sine,
        -l / 2 * neighborhood_vehicle_heading_cosine - -h / 2 * neighborhood_vehicle_heading_sine]).reshape(-1)
    neighborhood_vehicle_right_down_corner = relative_position + np.array([
        h / 2 * neighborhood_vehicle_heading_cosine + -l / 2 * neighborhood_vehicle_heading_sine,
        -l / 2 * neighborhood_vehicle_heading_cosine - h / 2 * neighborhood_vehicle_heading_sine]).reshape(-1)

    # corner
    if type == 'corner':
        return np.array([neighborhood_vehicle_left_up_corner,
                         neighborhood_vehicle_right_up_corner,
                         neighborhood_vehicle_left_down_corner,
                         neighborhood_vehicle_right_down_corner])

    return None


"""get neighborhood vehicle corresponding radius
"""
def get_neighborhood_vehicle_relative_radius(ego_position,
                                             neighborhood_vehicle_position,
                                             neighborhood_vehicle_bounding_box,
                                             neighborhood_vehicle_heading):
    neighborhood_vehicle_corner_relative_position = get_neighborhood_vehicle_relative_position(ego_position,
                                                                                               neighborhood_vehicle_position,
                                                                                               neighborhood_vehicle_bounding_box,
                                                                                               neighborhood_vehicle_heading,
                                                                                               type='center')
    
    neighborhood_vehicle_corner_relative_radius = np.sqrt(neighborhood_vehicle_corner_relative_position[0]**2
                                                         +neighborhood_vehicle_corner_relative_position[1]**2)
    return neighborhood_vehicle_corner_relative_radius


"""get self state
"""
def get_self_state(env_obs, args):
    ego_state = env_obs.ego_vehicle_state
                                                                                              #  shape | description
                                                                                              # ---------------------------
    ego_position         = ego_state.position[:2] / args.construct_obs.norm_position          #    2   |   [x, y]
    ego_bounding_box     = np.array([ego_state.bounding_box.length,                           #    2   |   [length, height]
                                     ego_state.bounding_box.width]) \
                                      / args.construct_obs.norm_bounding_box
    ego_heading          = np.array([ego_state.heading]) / args.construct_obs.norm_heading    #    1   |   [heading]
    ego_steering         = np.array([ego_state.steering])                                     #    1   |   [steering]
    ego_linear_velocity  = ego_state.linear_velocity[:2] / args.construct_obs.norm_velocity   #    2   |   [v_x, v_y]
    ego_angular_velocity = np.array([ego_state.angular_velocity[2]])                          #    1   |   [av_z]
    ego_line             = np.array([ego_state.lane_index])                                   #    1   |   [lance_index]

    ego_self_state       = np.concatenate([ego_position, 
                                           ego_bounding_box, 
                                           ego_heading, 
                                           ego_steering, 
                                           ego_linear_velocity, 
                                           ego_angular_velocity, 
                                           ego_line], axis=0)
    
    return ego_self_state


"""get distance from center(without abs)
"""
def get_distance_from_center(env_obs):
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]

    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return norm_dist_from_center


"""get heading error
"""
def get_heading_error(env_obs):
    indices = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 50])

    ego_wp_paths = env_obs.waypoint_paths
    ego_wps_len = [len(path) for path in ego_wp_paths]
    ego_max_len_lane_index = np.argmax(ego_wps_len)
    ego_max_len_lane = np.max(ego_wps_len)

    # 如果path的长度小于原先设定的indices系数，则需进行相应更改
    last_wp_index = 0
    for i, wp_index in enumerate(indices):
        if wp_index > ego_max_len_lane - 1:
            indices[i:] = last_wp_index
            break
        last_wp_index = wp_index

    ego_sample_wp_path = [ego_wp_paths[ego_max_len_lane_index][i] for i in indices]
    ego_heading_error = np.array([float(wp.relative_heading(env_obs.ego_vehicle_state.heading)) for wp in ego_sample_wp_path])
    
    return ego_heading_error

"""get heading position
"""
def get_heading_position(env_obs, args):
    indices = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 50])

    ego_wp_paths = env_obs.waypoint_paths
    ego_wps_len = [len(path) for path in ego_wp_paths]
    ego_max_len_lane_index = np.argmax(ego_wps_len)
    ego_max_len_lane = np.max(ego_wps_len)

    # 如果path的长度小于原先设定的indices系数，则需进行相应更改
    last_wp_index = 0
    for i, wp_index in enumerate(indices):
        if wp_index > ego_max_len_lane - 1:
            indices[i:] = last_wp_index
            break
        last_wp_index = wp_index
    
    ego_sample_wp_path = [ego_wp_paths[ego_max_len_lane_index][i] for i in indices]
    ego_heading_position = np.concatenate([wp.pos for wp in ego_sample_wp_path]) / args.construct_obs.norm_position
    return ego_heading_position


    
"""map neighborhood vehicle coordinates to corresponding quadrants and radius
"""
def map_neighborhood_vehicle_position_to_quadrant_radius(ego_position,                     
                                                         ego_heading,  
                                                         neighborhood_vehicle_position,
                                                         neighborhood_vehicle_bounding_box,  
                                                         neighborhood_vehicle_heading,  
                                                         lidar_ratio):  # num of lidar 
    neighborhood_vehicle_corner_relative_position = get_neighborhood_vehicle_relative_position(ego_position,
                                                                                               neighborhood_vehicle_position,
                                                                                               neighborhood_vehicle_bounding_box,
                                                                                               neighborhood_vehicle_heading)
    # change coordinate
    ego_heading = change_radian_normal_format(ego_heading)
    neighborhood_vehicle_corner_angle = np.arctan2(neighborhood_vehicle_corner_relative_position[:, 1],
                                                   neighborhood_vehicle_corner_relative_position[:, 0])
    neighborhood_vehicle_corner_relative_angle = neighborhood_vehicle_corner_angle - ego_heading


    lidar_sensitivity = 2 * np.pi / lidar_ratio
    neighborhood_vehicle_corner_quadrant_radius = []
    for neighborhood_vehicle_corner_relative_angle_, neighborhood_vehicle_corner_relative_position_ in zip(
            neighborhood_vehicle_corner_relative_angle, neighborhood_vehicle_corner_relative_position):

        neighborhood_vehicle_corner_relative_angle_ = set_angle_valid_scope(neighborhood_vehicle_corner_relative_angle_)
        neighborhood_vehicle_corner_relative_quadrant_ = neighborhood_vehicle_corner_relative_angle_ // lidar_sensitivity

        neighborhood_vehicle_corner_relative_radius_ = np.sqrt(neighborhood_vehicle_corner_relative_position_[0]**2
                                                              +neighborhood_vehicle_corner_relative_position_[1]**2)

        neighborhood_vehicle_corner_quadrant_radius.append([neighborhood_vehicle_corner_relative_quadrant_, 
                                                            neighborhood_vehicle_corner_relative_radius_])
    
    return neighborhood_vehicle_corner_quadrant_radius

# ====================================================================================
#    construction safety characteristic
# ====================================================================================
"""Calculate the curvature of the road ahead by waypath
"""
def get_curvature_via_waypoint(env_obs, args):
    indices = np.array([args.safety.curvature_foresight_1,
                        args.safety.curvature_foresight_2,
                        args.safety.curvature_foresight_3,
                        args.safety.curvature_foresight_4,
                        args.safety.curvature_foresight_5])

    ego_wp_paths = env_obs.waypoint_paths
    ego_wps_len = [len(path) for path in ego_wp_paths]
    ego_max_len_lane_index = np.argmax(ego_wps_len)
    ego_max_len_lane = np.max(ego_wps_len)

    # 如果path的长度小于原先设定的indices系数，则需进行相应更改
    last_wp_index = 0
    for i, wp_index in enumerate(indices):
        if wp_index > ego_max_len_lane - 1:
            indices[i:] = last_wp_index
            break
        last_wp_index = wp_index

    ego_sample_wp_path = [ego_wp_paths[ego_max_len_lane_index][i] for i in indices]
    ego_heading_error = np.array([float(wp.relative_heading(env_obs.ego_vehicle_state.heading)) for wp in ego_sample_wp_path])
    curvature = ego_heading_error.mean()

    return curvature

"""Calculate speed upper bound
"""
def set_speed_via_waypoint_curvature(env_obs, args):
    curvature = get_curvature_via_waypoint(env_obs, args)

    speed_level = 1.0
    if curvature < args.safety.curvature_threshold_level_1:
        speed_level = args.safety.speed_level_0
    elif curvature >= args.safety.curvature_threshold_level_1 and curvature < args.safety.curvature_threshold_level_2:
        speed_level = args.safety.speed_level_1
    else:
        speed_level = args.safety.speed_level_2

    return speed_level


"""get the distance with the car in heading (if have)
"""
def get_distance_to_front_car(env_obs, args):
    distance = 999999.

    ego_position = env_obs.ego_vehicle_state.position[:2]
    ego_heading = env_obs.ego_vehicle_state.heading

    if env_obs.neighborhood_vehicle_states is None:
        pass
    else:
        for neighborhood_vehicle_state in env_obs.neighborhood_vehicle_states:
            neighborhood_vehicle_position = neighborhood_vehicle_state.position[:2]
            relative_distance = np.linalg.norm(ego_position-neighborhood_vehicle_position)
            if relative_distance < args.safety.car_limit_distance and is_ahead(ego_position, ego_heading, neighborhood_vehicle_position):
                if relative_distance < distance:
                    distance = relative_distance

    return distance

"""calculate speed for front car
"""
def set_speed_via_front_car(env_obs, args):
    distance = get_distance_to_front_car(env_obs, args)

    speed_level = 1.0
    if distance == 999999.:
        return speed_level

    if distance >= args.safety.distance_level_1 and distance < args.safety.car_limit_distance:
        speed_level = args.safety.speed_level_1
    elif distance >= args.safety.distance_level_2 and distance < args.safety.distance_level_1:
        speed_level = args.safety.speed_level_2
    elif distance >= args.safety.distance_level_3 and distance < args.safety.distance_level_2:
        speed_level = args.safety.speed_level_3
    elif distance >= args.safety.distance_level_4 and distance < args.safety.distance_level_3:
        speed_level = args.safety.speed_level_4
    elif distance >= args.safety.distance_level_5 and distance < args.safety.distance_level_4:
        speed_level = args.safety.speed_level_5
    else:
        speed_level = args.safety.speed_level_6
    
    return speed_level


"""
"""
def get_threaten_distance(env_obs, args):
    ego_position = env_obs.ego_vehicle_state.position[:2]
    ego_heading = env_obs.ego_vehicle_state.heading
    ego_bounding_box = np.array([env_obs.ego_vehicle_state.bounding_box.length, env_obs.ego_vehicle_state.bounding_box.width])

    ego_corner_positions = get_ego_position(ego_position,
                                           ego_heading,
                                           ego_bounding_box,
                                           args, 
                                           type='threaten') # lu, ru, ld, rd

    ego_corner = [ego_corner_positions[0], ego_corner_positions[1], ego_corner_positions[3], ego_corner_positions[2]]
    ego_threaten_scope = Path(ego_corner)

    distance = 999999.
    if env_obs.neighborhood_vehicle_states is None:
        pass
    else:
        for neighborhood_vehicle_state in env_obs.neighborhood_vehicle_states:
            neighborhood_vehicle_position = neighborhood_vehicle_state.position[:2]
            neighborhood_vehicle_heading = neighborhood_vehicle_state.heading
            neighborhood_vehicle_bounding_box = np.array([neighborhood_vehicle_state.bounding_box.length, neighborhood_vehicle_state.bounding_box.width])
   
            neighborhood_vehicle_corner_positions = get_ego_position(neighborhood_vehicle_position,
                                                                     neighborhood_vehicle_heading,
                                                                     neighborhood_vehicle_bounding_box,
                                                                     args, 
                                                                     type='corner') # lu, ru, ld, rd

            for neighborhood_vehicle_corner_position in neighborhood_vehicle_corner_positions:
                is_in_threaten_scope = ego_threaten_scope.contains_point(neighborhood_vehicle_corner_position)
                if is_in_threaten_scope:
                    raw_distance = np.linalg.norm(ego_position-neighborhood_vehicle_corner_position)

                    if is_ahead(ego_position, ego_heading, neighborhood_vehicle_position): # 前方有车状态
                        delta_distance = raw_distance - args.safety.threaten_distance_limit_front
                    else: # 后方有车状态
                        delta_distance = raw_distance - args.safety.threaten_distance_limit_back
                    delta_distance = min(delta_distance, 0)

                    if distance > delta_distance:
                        distance = delta_distance
    
    return distance


def get_mini_distance_with_two_rect(rect1, rect2):
    line1 = LineString(rect1)
    line2 = LineString(rect2)
    return line1.distance(line2)

def get_distance_to_neighborhood_vehicle(env_obs, args):
    ego_state = env_obs.ego_vehicle_state
    ego_position = ego_state.position[:2] 
    ego_bounding_box = np.array([ego_state.bounding_box.length,
                                 ego_state.bounding_box.width])
    ego_heading = np.array([ego_state.heading]) 

    ego_corner = get_ego_position(ego_position, 
                                  ego_heading, 
                                  ego_bounding_box, 
                                  args)

    ego_line = LineString([ego_corner[0], 
                           ego_corner[1], 
                           ego_corner[2], 
                           ego_corner[3], 
                           ego_corner[0]])

    distance_to_neighborhood_vehicle = []
    for neighborhood_vehicle_state in env_obs.neighborhood_vehicle_states:
        neighborhood_vehicle_position = neighborhood_vehicle_state.position[:2]
        neighborhood_vehicle_heading = neighborhood_vehicle_state.heading
        neighborhood_vehicle_bounding_box = np.array([neighborhood_vehicle_state.bounding_box.length, neighborhood_vehicle_state.bounding_box.width])

        neighborhood_vehicle_corner = get_ego_position(neighborhood_vehicle_position, 
                                                       neighborhood_vehicle_heading, 
                                                       neighborhood_vehicle_bounding_box, 
                                                       args)
        
        neighborhood_vehicle_line = LineString([neighborhood_vehicle_corner[0], 
                                                neighborhood_vehicle_corner[1], 
                                                neighborhood_vehicle_corner[2], 
                                                neighborhood_vehicle_corner[3], 
                                                neighborhood_vehicle_corner[0]])
        
        distance = ego_line.distance(neighborhood_vehicle_line)
        distance_to_neighborhood_vehicle.append(distance)
    
    return distance_to_neighborhood_vehicle
# ====================================================================================
#    construction utils
# ====================================================================================

"""Set the radian value of the angle to the normal range
"""
def set_angle_valid_scope(angle):
    while angle < -np.pi:
        angle += np.pi * 2
    while angle > np.pi:
        angle -= np.pi * 2

    return angle

"""change radian dai format
"""
def change_radian_dai_format(relative_radian):
    if relative_radian > -np.pi and relative_radian < -np.pi / 2:
        relative_radian += 1.5 * np.pi 
    else:
        relative_radian += - 0.5 * np.pi
    return set_angle_valid_scope(relative_radian)

"""change radian normal format
"""
def change_radian_normal_format(relative_radian):
    if relative_radian > np.pi / 2 and relative_radian < np.pi:
        relative_radian += - 1.5 * np.pi
    else:
        relative_radian += + 0.5 * np.pi 
    return set_angle_valid_scope(relative_radian)

"""detect whether neighborhood vehicle is ahead
"""
def is_ahead(ego_position,                     
             ego_heading,  
             neighborhood_vehicle_position):
    relative_radian = np.arctan2(neighborhood_vehicle_position[1] - ego_position[1],
                                 neighborhood_vehicle_position[0] - ego_position[0])
    
    relative_radian = change_radian_dai_format(relative_radian)

    if np.abs(relative_radian - ego_heading) < (np.pi / 20) or np.abs(relative_radian - ego_heading) > (np.pi * 1.95):
        return True
    return False


# ====================================================================================
#    construction utils
# ====================================================================================
def construct_single_n_nearest_obs(env_obs, args):
     # ----------------------------------------
    # step 1: agent characteristics
    ego_self_state = get_self_state(env_obs, args)

    # ----------------------------------------
    # step 2: n-nearest characteristics
    ego_state = env_obs.ego_vehicle_state

    n = args.construct_obs.n
    ego_n_nearest = np.zeros((n * 5), dtype=np.float) # 2 + 2 + 1 + 1 + 1

    if env_obs.neighborhood_vehicle_states is None:
        pass
    else:
        distance_to_neighborhood_vehicle = get_distance_to_neighborhood_vehicle(env_obs, args)

        ego_n_nearest_dict = {}
        for neighborhood_vehicle_state, distance in zip(env_obs.neighborhood_vehicle_states, distance_to_neighborhood_vehicle):
            neighborhood_vehicle_position = neighborhood_vehicle_state.position[:2]

            neighborhood_vehicle_heading = np.array(neighborhood_vehicle_state.heading)
            neighborhood_vehicle_speed = np.array(neighborhood_vehicle_state.speed)

            neighborhood_vehicle_obs = np.zeros((5), dtype=np.float)
            neighborhood_vehicle_obs[:2] = neighborhood_vehicle_position / args.construct_obs.norm_position
            neighborhood_vehicle_obs[2] = neighborhood_vehicle_speed / args.construct_obs.norm_velocity
            neighborhood_vehicle_obs[3] = neighborhood_vehicle_heading / args.construct_obs.norm_heading
            neighborhood_vehicle_obs[4] = distance / args.construct_obs.norm_distance

            # 添加到dict中，用于排序
            ego_n_nearest_dict[distance] = neighborhood_vehicle_obs
        
        for i, radius in enumerate(sorted(ego_n_nearest_dict)):
            if i == n:
                break
   
            ego_n_nearest[i*5:(i+1)*5] = ego_n_nearest_dict[radius]

    # ----------------------------------------
    # step 3: distance from center(without abs)  
    ego_distance_from_center = get_distance_from_center(env_obs)
    ego_distance_from_center = np.array([ego_distance_from_center])
    
    # ----------------------------------------
    # step 4: heading error
    ego_heading_error = get_heading_error(env_obs)

    # ----------------------------------------
    # step 5: goal
    ego_goal = np.array(env_obs.ego_vehicle_state.mission.goal.positions[0])

    return {
        "ego_state": ego_self_state,
        "ego_n_nearest": ego_n_nearest,
        "ego_distance_from_center": ego_distance_from_center,
        "ego_heading_error": ego_heading_error,
        "ego_goal": ego_goal
    }

def observation_adapter(env_obs):
    args = get_args()
    return construct_single_n_nearest_obs(env_obs, args)

def reward_adapter(env_obs, env_reward):
    ego_state = env_obs.ego_vehicle_state
    ego_wp_paths = env_obs.waypoint_paths

    # 1. 环境奖励
    total_reward = env_reward * 1.0
    
    # # 2.离中心点距离
    distance_from_center = abs(get_distance_from_center(env_obs))
    total_reward += -0.1 * distance_from_center 

    # 3. 检测碰撞
    #      vehicle中心所对应的lane与该lane前方车辆(若存在)距离与相应阈值进行判断
    crash_penalty = 0.
    if env_obs.events.collisions != []:
        crash_penalty = -5.0
    total_reward += crash_penalty

    # 4  lane penalty
    off_road_penalty = 0.
    if env_obs.events.off_road:
        off_road_penalty = -5.0 
    total_reward += off_road_penalty 

    # 5 时间惩罚
    # time_penalty = -1.0 * args.construct_reward.time_penalty_coefficient
    # total_reward += time_penalty

    return total_reward

def action_adapter(model_action):
    assert len(model_action) == 3
    return np.asarray(model_action)

def new_action_adapter(model_action):
    assert model_action in [0, 1, 2, 3]

    if model_action == 0:
        target_speed = 20 
        lane_change = 0
    elif model_action == 1:
        target_speed = 0
        lane_change = 0
    elif model_action == 2:
        target_speed = 14.
        lane_change = 1
    elif model_action == 3:
        target_speed = 14.
        lane_change = -1
    
    return [target_speed, lane_change]

def info_adapter(reward, info):
    return info

# ==================================================
# Continous Action Space
# throttle, brake, steering
# ==================================================

# ACTION_SPACE = gym.spaces.Box(
#     low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
# )
ACTION_SPACE = gym.spaces.Discrete(4)
# ==================================================
# Observation Space
# This observation space should match the output of observation(..) below
# ==================================================
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "ego_state": gym.spaces.Box(low=-1e10, high=1e10, shape=(10,)),
        "ego_n_nearest": gym.spaces.Box(low=-1e10, high=1e10, shape=(20,)),
        "ego_distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_heading_error": gym.spaces.Box(low=-1e10, high=1e10, shape=(10,)),
        "ego_goal": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,))
    }
)

agent_interface = AgentInterface(
    max_episode_steps=None,
    waypoints=True,
    # neighborhood < 60m
    neighborhood_vehicles=NeighborhoodVehicles(radius=60),
    # OGM within 64 * 0.25 = 16
    #ogm=OGM(64, 64, 0.25),
    action=ActionSpaceType.LaneWithContinuousSpeed,
)

agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=new_action_adapter,
    info_adapter=info_adapter,
)