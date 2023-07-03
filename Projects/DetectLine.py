import cv2
import numpy as np




# define range of blue color in HSV
lower_blue = np.array([85, 50, 50])
upper_blue = np.array([100, 255, 255])

# define range of orange color in HSV
lower_orange = np.array([10, 75, 180])
upper_orange = np.array([30, 180, 255])


# define range of red color in HSV
#lower_red = np.array([0, 50, 195])
#upper_red = np.array([10, 255, 255])

#lower_dark_red = np.array([175, 140, 210])
#upper_dark_red = np.array([255, 255, 255])

# define range of white color in HSV
#lower_white = np.array([0, 0, 0])
#upper_white = np.array([0, 0, 255])

#hsv_ranges = [[lower_blue, upper_blue], [lower_orange, upper_orange], [lower_red, upper_red], [lower_dark_red, upper_dark_red]]
hsv_ranges = [[lower_orange, upper_orange]]
def get_center_of_mass(image):
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # create blank mask with size of original image with 1 channel
    mask = np.zeros_like(hsv_frame[:, :, 0], dtype=np.uint8)
    
    # loop through each range and bitwise or mask
    for lower, upper in hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))
    
    # apply erode to mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=4)
    
    # apply dilate to mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    mask_with_edges = cv2.Canny(mask, 100, 200)
    
    # detect contours
    contours, hierarchy = cv2.findContours(mask_with_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    paths = []
    # loop through each contour
    '''
    for contour in contours:
        # draw contour on mask
        cv2.drawContours(image, contour, -1, (0, 0, 255), 3)
        path = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        paths.append(path)
        cv2.drawContours(image, [path], -1, (0, 255, 0), 3)

    '''
    cv2.imshow("Mask", mask)
    cv2.imshow("HSV Frame", hsv_frame)
    #cv2.imshow("Mask with Edges", mask_with_edges)
    #cv2.imshow("Frame", image)
    
    cx = None
    cy = None
    #get center of mass from mask
    if len(contours) > 0:
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(image, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(image, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image, (cx, cy)
    
# segment the image into 5 equal horizontal sections
def segment_image(image, num_segments=5):
    height, width, _ = image.shape
    segment_height = int(height / num_segments)
    segments = []
    for i in range(num_segments):
        segments.append(image[i * segment_height : (i + 1) * segment_height, 0:width].copy())
    return segments

#combine all segments into one image with lines separating each segment
def combine_segments(segments):
    combined_image = np.zeros((segments[0].shape[0] * len(segments), segments[0].shape[1], 3), dtype=np.uint8)
    for i, segment in enumerate(segments):
        combined_image[i * segment.shape[0] : (i + 1) * segment.shape[0], 0:segment.shape[1]] = segment
        #cv2.line(combined_image, (0, (i + 1) * segment.shape[0]), (combined_image.shape[1], (i + 1) * segment.shape[0]), (255, 255, 255), 2)
    return combined_image
'''
# Open image from file
original_frame = cv2.imread("images/orange-paper.jpg")

# Rotate image
#original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)

# Resize image
original_frame = cv2.resize(original_frame, (640, 480))

segments = segment_image(original_frame, num_segments=7)
processed_segments = []
for i, segment in enumerate(segments):
    img = get_center_of_mass(segment.copy())
    #cv2.imshow(str(i), img[0])
    processed_segments.append(img[0])
    print(f"Segment {i} center of mass: {img[1]}")

combined_image = combine_segments(processed_segments)

img = get_center_of_mass(original_frame.copy())
cv2.imshow("Combined Image", img[0])

cv2.waitKey(0)
'''
