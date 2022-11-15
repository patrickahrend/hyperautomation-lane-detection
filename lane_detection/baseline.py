# Libraries Needed
import math
import time
import numpy as np
import cv2  # for image/video processing


#################################################################
# Steps needed for lane detection:                              #
# 1. Denoising the video frames using blurring kernel           #
# 2. Graysccaling and detecting edges on the frames with        #
#    Canny edge detection                                       #
# 3. Drawing area of interest to embed lanes on the video frame #
# 4. Perspective warping                                        #
# 5. Segmentation of lanes using vertical histogram projection  #
# 6. Detecting lines on the video frame using Hough Lines Polar #
#    and line optimization                                      #
# 7. Displaying lines on the frame                              #
# 8. Turn prediction                                            #
# 9. Whole process orchestrator                                 #
#################################################################

# STEP 1: Denoising the video frames using blurring
def denoise_frame(frame):
    # kernel = (3x3 matrix with 1/9 as values)
    kernel = np.ones((3, 3), np.float32) / 9
    # filter used for blurring, others can be used as well
    denoised_frame = cv2.filter2D(frame, -1, kernel)
    return denoised_frame


# STEP 2: Graysccaling and detecting edges on the frames with Canny edge detection
def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # make frame gray
    canny_edges = cv2.Canny(gray, 50, 150)  # draw edges
    return canny_edges


# additional aperture_size:
def detect_edges_with_aperture(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # make frame gray
    canny_edges = cv2.Canny(gray, 50, 150, apertureSize=5)  # more lines & more precise
    return canny_edges


# STEP 3: Drawing area of interest to embed lanes on the video frame
def region_of_interest(frame):
    # basically image is represented like array of numbers
    # we create a mask for it, where only road is represented with 1, and everything else with 0
    # that way when we do bitwise AND, only road edges will be shown
    height, width = frame.shape
    mask = np.zeros_like(frame)  # we created basically a black background (only 0)
    # now we need 1's to fill our road in order to create a mask:
    polygon = np.array([[
        (int(width * 0.30), height),  # Bottom-left point
        (int(width * 0.46), int(height * 0.72)),  # Top-left point
        (int(width * 0.58), int(height * 0.72)),  # Top-right point
        (int(width * 0.82), height),  # Bottom-right point
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)  # filling it with white color (255 is white)
    roi = cv2.bitwise_and(frame, mask)  # now we only have road edges represented
    return roi


# STEP 4: Perspective Warping
# Handle perspective changes by looking at the road from sky view angle
# so we take in normal frame and transform it into bird view perspective
def warp_perspective(frame):
    height, width = frame.shape
    offset = 50  # needed for later
    # Perspective points to be warped
    source_points = np.float32([
        [int(width * 0.46), int(height * 0.72)],  # Top-left point
        [int(width * 0.58), int(height * 0.72)],  # Top-right point
        [int(width * 0.30), height],  # Bottom-left point
        [int(width * 0.82), height]])  # Bottom-right point

    # Window to be shown
    destination_points = np.float32([
        [offset, 0],  # Top-left point
        [width - 2 * offset, 0],  # Top-right point
        [offset, height],  # Bottom-left point
        [width - 2 * offset, height]])  # Bottom-right point

    # now we have sky(bird) view coordinates which we apply using cv2 function
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    # finally we can warp
    skyview = cv2.warpPerspective(frame, matrix, (width, height))
    return skyview


# STEP 5: Segmentation of lanes using vertical histogram projection
# input: edged and skyview frame!!
def histogram(frame):
    histogram = np.sum(frame, axis=0)
    # reason we use histogram is because lane lines will show in it like peaks in the histogram
    # hence we can easily calculate midpoint and right/left line
    # Find mid point on histogram
    midpoint = np.int(histogram.shape[0] / 2)
    # Compute the left max pixels
    leftx_base = np.argmax(histogram[:midpoint])
    # Compute the right max pixels
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base


# STEP 6: Detecting lines on the video frame using Hough Lines Polar and line optimization
def detect_lines(frame):
    # PLACE FOR POSSIBLE OPTIMIZATION
    # 4th parameter -> treshold only lines with >treshhold aree returned
    # 5th parameter -> only lines with minLineLength or more get accepted
    # 6th parameter -> max allowed gap between points on the same line to link them
    line_segments = cv2.HoughLinesP(frame, 1, np.pi / 180, 20,
                                    np.array([]), minLineLength=40, maxLineGap=150)
    return line_segments  # Return line segment on road


# detect_lines method is not good bc it draws more lines close to eachother instead of only one
# we optimze this:
def optimize_lines(frame, lines):
    # we store both lines in one array, then left lines in left_fit and right lines in right_fit

    if lines is not None:
        lane_lines = []  # For both lines
        left_fit = []  # For left line
        right_fit = []  # For right line

    height, width, _ = frame.shape

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)  # get coordinates, (x1,y1) of one point, (x2,y2) of another
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # polyfit(x,y, degree) returns polynomial values from the coordinates and degree of the fitting polynomial
        # => we get slope and intercept
        slope = parameters[0]
        intercept = parameters[1]
        # now we can check to which lines array we want to add
        if slope < 0:  # Here we check the slope of the lines
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

        # check for left lines
        # we use map_coordinates to go back to the coordinates from slope and intercept
    if len(left_fit) > 0:
        # Averaging fits for the left line
        left_fit_average = np.average(left_fit, axis=0)
        # Add result of mapped points to the list lane_lines
        lane_lines.append(map_coordinates(frame, left_fit_average))

    # Here we ckeck whether fit for the right line is valid
    if len(right_fit) > 0:
        # Averaging fits for the right line
        right_fit_average = np.average(right_fit, axis=0)
        # Add result of mapped points to the list lane_lines
        lane_lines.append(map_coordinates(frame, right_fit_average))

    return lane_lines


def map_coordinates(frame, parameters):
    height, width, _ = frame.shape  # Take frame size
    slope, intercept = parameters  # Unpack slope and intercept from the given parameters

    if slope == 0:  # Check whether the slope is 0
        slope = 0.1  # handle it for reducing Division by Zero error

    y1 = height  # Point bottom of the frame
    y2 = int(height * 0.72)  # Make point from middle of the frame down
    x1 = int((y1 - intercept) / slope)  # Calculate x1 by the formula (y-intercept)/slope
    x2 = int((y2 - intercept) / slope)  # Calculate x2 by the formula (y-intercept)/slope

    return [[x1, y1, x2, y2]]  # Return point as array


##################################################################################################################
##################################################################################################################
#                           HARD WORK DONE; NOW WE NEED TO DISPLAY EVERYTHING
##################################################################################################################
##################################################################################################################
# STEP 7: Displaying lines on the frame

def display_lines(frame, lines):
    mask = np.zeros_like(frame)  # again we create a mask
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 5)  # draw the line from the coordinates on the mask

    # merge mask with the frame
    frame = cv2.addWeighted(frame, 0.8, mask, 1, 1)
    return frame


def display_heading_line(frame, up_center, low_center):
    heading_image = np.zeros_like(frame)  # again mask
    height, width, _ = frame.shape

    x1 = int(low_center)
    y1 = height
    x2 = int(up_center)
    y2 = int(height * 0.72)

    cv2.line(heading_image, (x1, y1), (x2, y2), (0, 0, 255), 5)  # draw the line
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)  # add it to our frame

    return heading_image


# STEP 8: TURN PREDICTION
# we make turn prediction using the central point between two lines, this is directly connected to our histograms
# important: central point is not static, it changes as the video changes
def get_floating_center(frame, lane_lines):
    height, width, _ = frame.shape  # Take frame size

    if len(lane_lines) == 2:  # Here we check if there is 2 lines detected
        left_x1, _, left_x2, _ = lane_lines[0][0]  # Unpacking left line
        right_x1, _, right_x2, _ = lane_lines[1][0]  # Unpacking right line

        # formula: M = (x1 + x2) / 2

        low_mid = (right_x1 + left_x1) / 2  # Calculate the relative position of the lower middle point
        up_mid = (right_x2 + left_x2) / 2

    else:  # Handling undetected lines
        up_mid = int(width * 1.9)
        low_mid = int(width * 1.9)

    return up_mid, low_mid


def add_text(frame, image_center, left_x_base, right_x_base):
    # text output
    lane_center = left_x_base + (right_x_base - left_x_base) / 2  # Find lane center between two lines
    deviation = image_center - lane_center  # Find the deviation

    if deviation > 160:  # Prediction turn according to the deviation
        text = "Smooth Left"
    elif deviation < 40 or deviation > 150 and deviation <= 160:
        text = "Smooth Right"
    elif deviation >= 40 and deviation <= 150:
        text = "Straight"

    cv2.putText(frame, "DIRECTION: " + text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)  # Draw direction

    return frame  # Retrun frame with the direction on it


# STEP 9: WE CALL EVERYTHING WE HAVE DEFINED

def process_frame(frame):
    # Declaring variables for fps
    avg_fps = 0
    fps_list = []

    start_time = time.time()  # Start the timer

    edges = detect_edges(frame)
    denoised_frame = denoise_frame(frame)  # Denoise frame from artifacts
    canny_edges = detect_edges(denoised_frame)  # Find edges on the frame
    roi_frame = region_of_interest(canny_edges)  # Draw region of interest
    warped_frame = warp_perspective(canny_edges)  # Warp the original frame, make it skyview
    left_x_base, right_x_base = histogram(warped_frame)  # Take x bases for two lines
    lines = detect_lines(roi_frame)  # Detect lane lines on the frame
    lane_lines = optimize_lines(frame, lines)  # Optimize detected line
    mul_lines = display_lines(frame, lines)
    lane_lines_image = display_lines(frame, lane_lines)  # Display solid and optimized lines

    up_center, low_center = get_floating_center(frame, lane_lines)  # Calculate the center between two lines

    heading_line = display_heading_line(lane_lines_image, up_center, low_center)

    final_frame = add_text(heading_line, low_center, left_x_base, right_x_base)  # Predict and draw turn

    fps = round(1.0 / (time.time() - start_time), 1)  # Here we calculate the fps
    fps_list.append(fps)  # Append fps to fps list
    cv2.putText(final_frame, "FPS: " + str(fps), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)  # Draw FPS



    return final_frame  # Return final frame
