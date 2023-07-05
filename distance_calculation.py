import cv2
import numpy as np
import time
import math

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

# CAMERA PARAMETERS
FOCAL_LENGTH = 1000.0 #mm
HFOVANGLE = 71.2 #degrees
VFOVANGLE = 40.5 #degrees
VFOVANGLEMID = 15.75 #degrees
RESOLUTION_X = 1280 #px
RESOLUTION_Y = 720 #px

# OBJECT PARAMETERS
KNOWN_WIDTH = 40.00 #mm

# Returns distance in a straight line (not accounting for any angle) to the object. Not super useful
def getLineDistance(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width

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


# def lawOfCosines3rdSide(side1, side2, angle):
#     C = math.radians(C)
    
#     # Calculate the third side using the law of cosines
#     c = math.sqrt(a**2 + b**2 - 2*a*b*math.cos(C))
    
#     return c

ball_data = []
# Returns gap in Z axis between 2 balls
def getZGap(ball1, ball2):
    angle = abs(ball1.vangle-ball2.vangle)
    x = ((min(ball1.vertical_distance,ball2.vertical_distance))*(math.sin(math.radians(angle))))
    if ball1.y < ball2.y:
        return x*-1
    return x

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

    cv2.imshow("Detected Object", image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        if len(ball_data) < 2:
            ball_data.append(ball)
            print(f"Picture {len(ball_data)} captured.")
        else:
            print(ball_data[0])
            print(ball_data[1])
            print(getZGap(ball_data[0], ball_data[1]))

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()