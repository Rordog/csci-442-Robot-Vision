import math

import cv2 as cv
import numpy as np
import pupil_apriltags as apriltag
from scipy.spatial.transform import Rotation as R
import pyttsx3

from PWMController import PWMController
from USBController import USBController

usb_controller = USBController()
throttle = PWMController(0, usb_controller)
turn = PWMController(1, usb_controller)
head_tilt = PWMController(4, usb_controller)
head_twist = PWMController(3, usb_controller)

face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

engine = pyttsx3.init()


detector = apriltag.Detector(decode_sharpening=0.5)


def tilt_head_up(a, b, c):
	head_tilt.set(0.1)
	return True

def tilt_head_down(a, b, c):
	head_tilt.set(1)
	return True

def look_right(a, b, c):
        head_twist.set(1)
        return True

def look_ahead(a, b, c):
        head_twist.set(0)
        return True

def get_orient_to_target_by_id(tag_id):
	def orient_to_target(color_image, depth_image, global_memory):
	    cv.namedWindow("Orient to target")
	    rotation_threshold = 5 # degrees
	        
	    # Convert BGR to Grayscale uint8
	    gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
	    
	    cx = color_image.shape[1] / 2
	    cy = color_image.shape[0] / 2
	        
	    # Detect AprilTags (returns an array of detection results)
	    results = detector.detect(gray, estimate_tag_pose=True, camera_params=(color_image.shape[1], color_image.shape[0], cx, cy), tag_size=0.05)
	    img = color_image.copy()
	    position = None
	    # Find the result with tag ID 0
	    for result in results:
	        #draw circle around the tag
	        cv.circle(img, (int(result.center[0]), int(result.center[1])), 5, (0, 255, 0), -1)
	        # Label the center of the tag
	        cv.putText(img, str(result.tag_id), (int(result.center[0]), int(result.center[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	        if result.tag_id == tag_id:
	            position = result.center
	    #print("Position: " + str(position))
	    cv.imshow("Orient to target", img)
	    if position is None:
	        # Spin until the target is in the center of the camera
	        #print("No target")
	        turn.set(.435)
	        return False
	    #print("Difference: " + str(position[0] - cx))
	    if abs(position[0] - cx) < rotation_threshold:
	        turn.set(0.0)
	        #print("Locked On target")
	        return True
	    
	    turn_speed = calculate_turn_speed(position[0], max_value=0.25,  max_pixel=color_image.shape[1])
	    turn.set(turn_speed)
	    return False
	return orient_to_target

def orient_to_field(color_image, depth_image, global_memory):
    
    rotation_threshold = 10 # degrees
    
    # Convert BGR to Grayscale uint8
    gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    
    cx = color_image.shape[1] / 2
    cy = color_image.shape[0] / 2
    
    # Detect AprilTags (returns an array of detection results)
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=(color_image.shape[1], color_image.shape[0], cx, cy), tag_size=0.05)
    
    rotation = None
    # Find the result with tag ID 0
    for result in results:
        if result.tag_id == 0:
            rotation = R.from_matrix(result.pose_R).as_euler('xyz', degrees=True)
            break
    #print("Rotation: " + str(rotation))            
    if rotation is None:
        # Spin until the target is in the center of the camera
        turn.set(0.435)

        return False

    if abs(rotation[1]) < rotation_threshold:
        turn.set(0.0)
        return True

    if rotation[1] > 0:
        turn.set(-0.435)
    else:
        turn.set(0.435)

    return False

def get_cross_field_by_id(tag_id, distance):
	def cross_field(color_image, depth_image, global_memory):
	   threshold = distance # meters
	        
	   # Convert BGR to Grayscale uint8
	   gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
	        
	   cx = color_image.shape[1] / 2
	   cy = color_image.shape[0] / 2
	        
	   # Detect AprilTags (returns an array of detection results)
	   results = detector.detect(gray, estimate_tag_pose=True, camera_params=(color_image.shape[1], color_image.shape[0], cx, cy), tag_size=0.159)
	   img = color_image.copy()
	   position = None
	   # Find the result with tag ID 0
	   for result in results:
	        #draw circle around the tag
	        cv.circle(img, (int(result.center[0]), int(result.center[1])), 5, (0, 255, 0), -1)
	        # Label the center of the tag
	        cv.putText(img, str(result.tag_id), (int(result.center[0]), int(result.center[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	        if result.tag_id == tag_id:
	            position = result.pose_t
	  # print("Position: " + str(position))
	   cv.imshow("Distance", img)
	   if position is None:
	        # Spin until the target is in the center of the camera
	   #     print("No target")
	        throttle.set(0.0)
	        return False
	  # print("Distance")
	   if abs(position[2]) > threshold:
	        throttle.set(0.5)
	   #     print("Locked On target")
	        return False
	   else:
	        throttle.reset()
	        return True 
	    
	    #turn_speed = calculate_turn_speed(position[0], max_value=0.3,  max_pixel=color_image.shape[1])
	    #turn.set(turn_speed)
	   return False
	return cross_field

def get_say_function(text):
    def func(color_image, depth_image, global_memory):
        #engine.say(text)
        #engine.runAndWait()
        print("ROBOT SAYS: " + text)
        return True
    return func

def say_color(a, b, c):
    print("ROBOT SAYS: I SEE " + c["ICE_COLOR"])
    return True

def drive_back(color_image, depth_image, global_memory):
   threshold = 2.5 # meters
        
   # Convert BGR to Grayscale uint8
   gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        
   cx = color_image.shape[1] / 2
   cy = color_image.shape[0] / 2
        
   # Detect AprilTags (returns an array of detection results)
   results = detector.detect(gray, estimate_tag_pose=True, camera_params=(color_image.shape[1], color_image.shape[0], cx, cy), tag_size=0.159)
   img = color_image.copy()
   position = None
   # Find the result with tag ID 0
   for result in results:
        #draw circle around the tag
        cv.circle(img, (int(result.center[0]), int(result.center[1])), 5, (0, 255, 0), -1)
        # Label the center of the tag
        cv.putText(img, str(result.tag_id), (int(result.center[0]), int(result.center[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if result.tag_id == 0:
            position = result.pose_t
   #print("Position: " + str(position))
   cv.imshow("Distance", img)
   if position is None:
        # Spin until the target is in the center of the camera
    #    print("No target")
        throttle.set(-0.5)
        return False
  # print("Distance")
   if abs(position[2]) < threshold:
        throttle.set(-0.5)
   #     print("Locked On target")
        return False
   else:
        throttle.reset()
        return True 
    
    #turn_speed = calculate_turn_speed(position[0], max_value=0.3,  max_pixel=color_image.shape[1])
    #turn.set(turn_speed)
   return False 

def find_human(color_image, depth_image, global_memory):
    gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    return len(faces) > 0

def go_to_human(color_image, depth_image, global_memory):
    pass

# Orange HSV range
lower_orange = np.array([5, 180, 80])
upper_orange = np.array([25, 255, 255])

# Green HSV range
lower_green = np.array([40, 100, 75])
upper_green = np.array([70, 255, 255])

# Pink HSV range
lower_pink = np.array([160, 100, 100])
upper_pink = np.array([185, 255, 255])

# Red HSV range
lower_red = np.array([0, 150, 195])
upper_red = np.array([10, 255, 255])

def read_color(color_image, depth_image, global_memory):
    # Convert BGR to HSV
    hsv = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)
    small_hsv = cv.resize(hsv, (320, 240))
    cv.imshow("HSV", small_hsv)
    # Threshold the HSV image to get only colors
    mask_orange = cv.inRange(hsv, lower_orange, upper_orange)
    mask_green = cv.inRange(hsv, lower_green, upper_green)
    mask_pink = cv.inRange(hsv, lower_pink, upper_pink)
    cv.imshow("Orange Mask", mask_orange)
    #cv.imshow("Green Mask", mask_green)
    #cv.imshow("Pink Mask", mask_pink)
    # Find the contours of the colored objects
    contours_orange, hierarchy_orange = cv.findContours(mask_orange, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_green, hierarchy_green = cv.findContours(mask_green, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_pink, hierarchy_pink = cv.findContours(mask_pink, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour_orange = None
    largest_contour_green = None
    largest_contour_pink = None
    largest_area_orange = 0
    largest_area_green = 0
    largest_area_pink = 0

    for contour in contours_orange:
        area = cv.contourArea(contour)
        if area > largest_area_orange:
            largest_area_orange = area
            largest_contour_orange = contour

    for contour in contours_green:
        area = cv.contourArea(contour)
        if area > largest_area_green:
            largest_area_green = area
            largest_contour_green = contour

    for contour in contours_pink:
        area = cv.contourArea(contour)
        if area > largest_area_pink:
            largest_area_pink = area
            largest_contour_pink = contour

    # Determine the largest contour of the three colors
    largest_contour = None
    largest_area = 0
    if largest_area_orange > largest_area_green and largest_area_orange > largest_area_pink:
        largest_contour = largest_contour_orange
        largest_area = largest_area_orange
        global_memory['ICE_COLOR'] = 'ORANGE'
    elif largest_area_green > largest_area_orange and largest_area_green > largest_area_pink:
        largest_contour = largest_contour_green
        largest_area = largest_area_green
        global_memory['ICE_COLOR'] = 'GREEN'
    elif largest_area_pink > largest_area_orange and largest_area_pink > largest_area_green:
        largest_contour = largest_contour_pink
        largest_area = largest_area_pink
        global_memory['ICE_COLOR'] = 'PINK'
    #print(global_memory['ICE_COLOR'])
    # If the largest contour is large enough, return True
    return largest_area > 10000
    #return False    



def cross_field_back(color_image, depth_image, global_memory):
    pass

def hit_target(color_image, depth_image, global_memory):
    target_color = global_memory['ICE_COLOR']
    if target_color == 'ORANGE':
        lower = lower_orange
        upper = upper_orange
    elif target_color == 'GREEN':
        lower = lower_green
        upper = upper_green
    elif target_color == 'PINK':
        lower = lower_pink
        upper = upper_pink
        
    # Convert BGR to HSV
    hsv = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only colors
    mask = cv.inRange(hsv, lower, upper)
    
    # Find center of the target
    M = cv.moments(mask)
    if M['m00'] != 0:
        cx = int(M["m10"] / M["m00"])
        turn.set(calculate_turn_speed(cx, max_value=0.5, max_pixel=color_image.shape[1]))
        throttle.set(0.45)
    else:
        turn.set(0.0)
        throttle.set(0.45)
        
    distance = get_median_distance_from_mask(mask, depth_image)
    return False

def get_median_distance_from_mask(mask, depth_image):
    # Get the depth values from the mask
    depth_values = depth_image[mask > 0]
    
    # Get the median depth value
    median_depth = np.median(depth_values)
    
    return median_depth

def calculate_turn_speed(cx, max_value=0.5, max_pixel=640, min_pixel=0, min_turning_speed = 0.45):
    if cx is None:
        return 0.0
    if cx >= max_pixel:
        return 0.0
    if cx <= min_pixel:
        return 0

    # Linearly map the pixel value to a value in the range of with 0 being the center of the image
    place_in_image = (cx - min_pixel) / (max_pixel - min_pixel)
    first_pass = (place_in_image - 0.5) * 2 * max_value
    return math.copysign(min_turning_speed,  first_pass) + first_pass

