
import pyrealsense2 as rs
import numpy as np
import cv2

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
trackArray = np.asanyarray(trackFrame.get_data())
tracker = cv2.TrackerKCF_create()
bbox = (240,240,280,280)

ok = tracker.init(trackArray, bbox)


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        tracker_frame = np.zeros(shape=[512,1280,3], dtype=np.uint8)
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        ok, bbox = tracker.update(color_image)
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(color_image, p1, p2, (255,0,0), 2, 1)
            centerX = (p1[0] + p2[0])/2
            centerY = (p1[1] + p2[1])/2
            zDepth = depth_frame.get_distance(int(centerX),int(centerY))
            b1 = (int(centerX-50)*2, (256 -(int(zDepth*100)-1)))
            b2 = (int(centerX+50)*2, (256 -(int(zDepth*100)+1)))
            print(f"{zDepth=}, {b1=}, {b2=}")
            cv2.rectangle(tracker_frame, b1, b2, (255,0,0), -1)
        else :
            # Tracking failure
            cv2.putText(color_image, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        cv2.rectangle(tracker_frame, (638, 254), (642, 258),(0,0,255), -1)
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        stacked = np.vstack((images, tracker_frame))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', stacked)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()
    
