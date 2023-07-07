import cv2
import numpy as np
import time
import math

# CAMERA PARAMETERS
FOCAL_LENGTH = 975.0 #mm
HFOVANGLE = 71.2 #degrees
VFOVANGLE = 40.5 #degrees
VFOVANGLEMID = 15.75 #degrees
RESOLUTION_X = 1280 #px
RESOLUTION_Y = 720 #px

# OBJECT PARAMETERS
KNOWN_WIDTH = 40.00 #mm

class Ball:
    def __init__(self, center_x, center_y, diameter, hangle, vangle, line_distance, vertical_distance, horizontal_distance, direct_distance):
        self.x = center_x
        self.y = center_y
        self.diameter = diameter
        self.hangle = hangle
        self.vangle = vangle
        self.line_distance = line_distance
        self.vertical_distance = vertical_distance
        self.horizontal_distance = horizontal_distance
        self.direct_distance = direct_distance

    def __str__(self):
        return f"Ball Data:\nx: {self.x},\ny: {self.y},\ndiameter: {self.diameter},\nhangle: {self.hangle},\nvangle: {self.vangle},\nline_distance: {self.line_distance},\nvertical_distance: {self.vertical_distance},\nhorizontal_distance: {self.horizontal_distance},\ndirect_distance: {self.direct_distance}\n"

possibilities = ["sports ball", "donut", "orange"]
ball_data = []

# Returns distance in a straight line (not accounting for any angle) to the object.
def getLineDistance(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width

def getActualLine(dist, vangle):
    return (dist*(math.cos(math.radians(abs(vangle)))))

# Center of x axis (assumes it just splits x axis in two)
def get_mid_x():
    return RESOLUTION_X/2

# Center of y axis
def get_mid_y():
    return (VFOVANGLEMID/VFOVANGLE)*RESOLUTION_Y

# Returns horizontal angle from middle of camera (imagine the normal angle being middle of camera)
def getHorizontalAngle(x_coordinate):
    midx = get_mid_x()
    halfangle = HFOVANGLE/2
    distfrommid = (x_coordinate-midx)
    angle = (distfrommid/midx)*halfangle
    return angle

# Returns vertical angle of object from normal (normal angle is at the HFOVANGLEMID)
def getVerticalAngle(y_coordinate):
    midy = get_mid_y()
    distfrommid = (midy-y_coordinate)
    angle = (distfrommid/RESOLUTION_Y)*VFOVANGLE
    return angle

# Returns direct distance accounting for angle to the object, accounting for both angles
def getDirectDistance(hangle, vangle, dist):
    x = (dist/(math.cos(math.radians(hangle))))
    return (x/(math.cos(math.radians(vangle))))

# Returns distance only accounting for horizontal angle
def getHorizontalDistance(hangle, dist):
    return (dist/(math.cos(math.radians(hangle))))

# Returns distance only accounting for vertical angle
def getVerticalDistance(vangle, dist):
    return (dist/(math.cos(math.radians(vangle))))

def lawOfCosines3rdSide(side1, side2, angle):    
    return (math.sqrt(side1**2 + side2**2 - 2*side1*side2*math.cos(math.radians(angle))))

# Angle opposite to side3
def lawOfCosinesAngle(side1, side2, side3):
    return (math.acos((side1**2 + side2**2 - side3**2) / (2 * side1 * side2)) * (180 / math.pi))

# Returns gap in Z axis between 2 balls
def getZGap(ball1, ball2):
    l1 = (ball1.vertical_distance)*(math.sin(math.radians(ball1.vangle)))
    l2 = (ball2.vertical_distance)*(math.sin(math.radians(ball2.vangle)))
    if (ball1.vangle*ball2.vangle > 0): #same sign
        x = abs(l1-l2)
    else:
        x = abs(l1) + abs(l2)
    if ball1.y < ball2.y:
        return x*-1
    return x

# Returns gap in X axis between 2 balls
def getXGap(ball1, ball2):
    l1 = (ball1.horizontal_distance)*(math.sin(math.radians(ball1.hangle)))
    l2 = (ball2.horizontal_distance)*(math.sin(math.radians(ball2.hangle)))
    if (ball1.hangle*ball2.hangle > 0): #same sign
        x = abs(l1-l2)
    else:
        x = abs(l1) + abs(l2)
    if ball1.x > ball2.x:
        return x*-1
    return x

def getYGap(ball1, ball2):
    x = abs(ball1.line_distance - ball2.line_distance)
    if (ball1.diameter > ball2.diameter):
        return  x*-1
    return x

# Returns gap in X and Y axis between 2 balls. Refer to getXandYGap.jpg on the method
def getXandYGap(ball1, ball2):
    # smallerside = 0
    if (ball1.horizontal_distance < ball2.horizontal_distance):
        smallerside = ball1.horizontal_distance
        smallersideangle1 = abs(ball1.hangle)
        longerside = ball2.horizontal_distance
        longersideangle1 = abs(ball2.hangle)
    else:
        smallerside = ball2.horizontal_distance
        smallersideangle1 = abs(ball2.hangle)
        longerside = ball1.horizontal_distance
        longersideangle1 = abs(ball1.hangle)
    inner_angle = abs(ball1.hangle - ball2.hangle)
    thirdside = lawOfCosines3rdSide(smallerside, longerside, inner_angle)
    smallersideangle2 = lawOfCosinesAngle(smallerside, thirdside, longerside)
    longersideangle2 = lawOfCosinesAngle(longerside, thirdside, smallerside)
    # print(f"""\n\n\nSmaller side:{smallerside}\nsmaller side angle1:{smallersideangle1}\nthird side:{thirdside}\nlonger side:{longerside}\n
    # longer side angle1:{longersideangle1}\n, innerangle: {inner_angle}\nsmallersideangle2: {smallersideangle2}\nlongersideangle2: {longersideangle2}""")
    smallersideangle3 = 180 - smallersideangle1 - smallersideangle2
    longersideangle3 = 90 - longersideangle1 - longersideangle2
    xgap = thirdside*(math.sin(math.radians(smallersideangle3)))
    ygap = thirdside*(math.cos(math.radians(smallersideangle3)))
    # print(f"\n\nsmaller side angle: {smallersideangle3}\nlonger side angle: {longersideangle3}\n xgap: {xgap}\nygap: {ygap}\n\n\n")
    # print(smallersideangle3, longersideangle3, thirdside)
    if (ball1.x > ball2.x):
        xgap *= -1
    if (ball1.diameter > ball2.diameter):
        ygap *= -1
    return(xgap,ygap)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the class labels and generate colors for each class
labels_path = "coco.names"
LABELS = open(labels_path).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

cap = cv2.VideoCapture(0)
time.sleep(0.4)

while True:
    ret, image = cap.read()

    if not ret:
        print("Error capturing image from webcam.")
        break

    # Prepare the image for inference
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform inference
    layer_names = net.getLayerNames()
    out_layers_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i[0] - 1] if type(i) == list else layer_names[i - 1] for i in out_layers_indices]
    outputs = net.forward(output_layers)

    # Initialize variables to store the object with the highest confidence
    highest_confidence = 0
    highest_conf_object = None

    midx = get_mid_x()
    midy = get_mid_y()
    cv2.line(image, (0, int(midy)), (RESOLUTION_X, int(midy)), (0, 255, 0), 2)            
    cv2.line(image, (int(midx), 0), (int(midx), RESOLUTION_Y), (0, 255, 0), 2)

    # Process the detection results
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > highest_confidence and LABELS[class_id] in possibilities:  # Check if the current object has a higher confidence
                highest_confidence = confidence
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                highest_conf_object = (x, y, w, h)

    # Draw the bounding box for the object with the highest confidence
    if highest_conf_object is not None:
        x, y, w, h = highest_conf_object
        diameter = int((w + h) / 2)

        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        radius = int((w + h) / 4)
        hangle = getHorizontalAngle(center_x)
        vangle = getVerticalAngle(center_y)
        line_distance = getLineDistance(KNOWN_WIDTH, FOCAL_LENGTH, diameter)/25.4
        direct_distance = getDirectDistance(hangle, vangle, line_distance)
        vertical_distance = getVerticalDistance(vangle, line_distance)
        horizontal_distance = getVerticalDistance(hangle, line_distance)
        ball = Ball(center_x, center_y, diameter, hangle, vangle, line_distance, vertical_distance, horizontal_distance, direct_distance)
        cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)

        cv2.putText(image, f"Line Distance: {line_distance:.2f} in", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Diameter: {diameter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Horizontal Angle: ({hangle: .2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Vertical Angle: ({vangle: .2f})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Vertical Distance: {vertical_distance:.2f} in", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Horizontal Distance: {horizontal_distance:.2f} in", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Direct Distance: {direct_distance:.2f} in", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"(X, Y): {center_x, center_y} in", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Actual Distance: {getActualLine(line_distance, vangle):.2f} in", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Detected Object", image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        if len(ball_data) < 2:
            ball_data.append(ball)
            print(f"Picture {len(ball_data)} captured.")
        else:
            print(ball_data[0])
            print(ball_data[1])
            xygap = getXandYGap(ball_data[0], ball_data[1])
            print("\n\nX Gap: ", xygap[0])
            print("Y Gap: ", xygap[1])
            print("Z Gap: ", getZGap(ball_data[0], ball_data[1]))
            print(getXGap(ball_data[0], ball_data[1]))
            print(getYGap(ball_data[0], ball_data[1]))
            print(getActualLine(line_distance, vangle))
            ball_data = []

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()