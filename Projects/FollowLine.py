from DetectLine import get_center_of_mass, segment_image, combine_segments

import pyrealsense2 as rs
import numpy as np
import cv2
from PWMController import PWMController
from USBController import USBController

def calculate_turn_speed(cx, max_value=0.5, max_pixel=640, min_pixel=0):
    if cx is None:
        return 0.0
    if cx >= max_pixel:
        return 0.0
    if cx <= min_pixel:
        return 0
    
    # Linearly map the pixel value to a value in the range of with 0 being the center of the image
    place_in_image = (cx - min_pixel) / (max_pixel - min_pixel)
    return (place_in_image - 0.5) * 2 * max_value


def calculate_throttle_speed(cy, max_value=0.5, max_pixel=480, min_pixel=0):
    if cy is None:
        return 0.0
    cy = max_pixel - cy

    if cy >= max_pixel:
        return 0.0
    if cy <= min_pixel:
        return 0
    
    # Linearly map the pixel value to a value in the range of with 0 being the center of the image
    place_in_image = (cy - min_pixel) / (max_pixel - min_pixel)
    return (place_in_image) * max_value


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

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
pipeline.start(config)

frame = pipeline.wait_for_frames()
trackFrame = frame.get_color_frame()
#trackArray = np.asanyarray(trackFrame.get_data())
#tracker = cv2.TrackerKCF_create()
#bbox = (240, 240, 280, 280)

#ok = tracker.init(trackArray, bbox)

try:
    usb_controller = USBController()
    throttle = PWMController(0, usb_controller)
    turn = PWMController(1, usb_controller)
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        image_segmented = segment_image(color_image, num_segments=5)

        combined = combine_segments(image_segmented[-3:])
        # Get center of mass
        img, CoM = get_center_of_mass(combined)

        # Calculate turn speed
        turn_speed = calculate_turn_speed(CoM[0], max_value=0.75, max_pixel=img.shape[1])

        # Calculate throttle speed
        throttle_speed = min(0.5, calculate_throttle_speed(CoM[1], max_value=0.75, max_pixel=img.shape[0]) + .1)

        print("Turn: " + str(turn_speed) + " Throttle: " + str(throttle_speed))
        turn.set(turn_speed)
        throttle.set(throttle_speed)

        # Show images
        cv2.namedWindow('Color Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Color Image', img)

        # Convert images to numpy arrays
        #color_image = np.asanyarray(color_frame.get_data())
        #color_colormap_dim = color_image.shape

        
        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        #cv2.imshow('RealSense', trackFrame)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()
    turn.reset()
    throttle.reset()

