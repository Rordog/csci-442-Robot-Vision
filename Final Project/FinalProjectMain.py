import StateFunctions
from DetectLine import get_center_of_mass, segment_image

import pyrealsense2 as rs
import numpy as np
import cv2
from PWMController import PWMController
from USBController import USBController
from StateFunctions import *


# Robot States
#robot_states = ["TILT_DOWN", "ORIENT",  "TILT_UP"][:2]
robot_states = ["TILT_UP", "LOOK_AHEAD", "ORIENT", "CROSS_FIELD", "SAY_MINING", "LOOK_RIGHT", "FIND_HUMAN", "ASK_FOR_ICE", "READ_COLOR", "SAY_COLOR", "LOOK_AHEAD", "ORIENT_BACK", "CROSS_FIELD_BACK", "SAY_GOAL_AREA", "TILT_DOWN", "HIT_TARGET"]
#robot_states = ["TILT_UP", "LOOK_RIGHT", "FIND_HUMAN", "LOOK_AHEAD"]
#robot_states = ["ORIENT", "CROSS_FIELD", "FIND_HUMAN","GO_TO_HUMAN", "READ_COLOR", "CROSS_FIELD_BACK", "HIT_TARGET"]
robot_states_dictionary = {"ORIENT": StateFunctions.get_orient_to_target_by_id(1), 
                           "ORIENT_BACK": StateFunctions.get_orient_to_target_by_id(0),
                           "CROSS_FIELD": StateFunctions.get_cross_field_by_id(1, 1.0),
			   "CROSS_FIELD_BACK": StateFunctions.get_cross_field_by_id(0, 2.0), 
                           "FIND_HUMAN": StateFunctions.find_human, 
                           "GO_TO_HUMAN": StateFunctions.go_to_human, 
                           "READ_COLOR": StateFunctions.read_color, 
                           "HIT_TARGET": StateFunctions.hit_target,
			   "TILT_DOWN": StateFunctions.tilt_head_down,
			   "TILT_UP": StateFunctions.tilt_head_up,
                           "SAY_COLOR": StateFunctions.say_color,
                           "ASK_FOR_ICE": StateFunctions.get_say_function("GIVE ME THE ICE COLOR"),
                           "SAY_MINING": StateFunctions.get_say_function("ENTERING DIGGING AREA"),
                           "SAY_GOAL_AREA": StateFunctions.get_say_function("ENTERING GOAL AREA"),
                           "FIND_HUMAN": StateFunctions.find_human,
                           "LOOK_AHEAD": StateFunctions.look_ahead,
                           "LOOK_RIGHT": StateFunctions.look_right}
robot_state_index = 0
global_memory = dict()

def run_robot_iteration(color_image, depth_image):
    global robot_state_index
    # Check robot state
    if robot_states_dictionary[robot_states[robot_state_index]](color_image, depth_image, global_memory):
        robot_state_index += 1
        if robot_state_index >= len(robot_states):
            return True
    return False





# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

device.query_sensors()[1].set_option(rs.option.exposure, 70)

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
prof = pipeline.start(config)



frame = pipeline.wait_for_frames()
trackFrame = frame.get_color_frame()
# trackArray = np.asanyarray(trackFrame.get_data())
# tracker = cv2.TrackerKCF_create()
# bbox = (240, 240, 280, 280)

# ok = tracker.init(trackArray, bbox)

try:

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()



        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not depth_frame or not color_frame:
            continue


        #print("color: "+ str(color_frame))
        #print("depth: "+ str(depth_frame))

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if run_robot_iteration(color_image, depth_image):
                print("DONE")
                break

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', color_image)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()
    turn.reset()
    throttle.reset()




    
