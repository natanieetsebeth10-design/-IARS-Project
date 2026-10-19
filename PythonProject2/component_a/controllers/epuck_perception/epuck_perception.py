"""
Component A - Step 1: Sensor Simulation and Perception Pipeline
=================================================================
Controller for the e-puck robot in component_a_world.wbt.

What this script does:
  1. Reads the 8 IR distance sensors (our "range scan" modality).
  2. Reads the camera image (our vision modality).
  3. Runs a simple OpenCV colour-threshold + contour detector on the
     camera image to find obstacle blobs (proxy for "landmark/obstacle
     detection" requirement).
  4. Logs a synchronised timestamped record (distance readings +
     detected obstacle count/positions) to data/perception_log.csv
     so the offline Python/Matplotlib scripts (Day 2-3) can do the
     sensor fusion, occupancy grid construction, and visualisation
     without needing Webots running at the same time.

Run this by placing it in:
  component_a/controllers/epuck_perception/epuck_perception.py
and setting the E-puck node's `controller` field to "epuck_perception".

Dependencies: numpy, opencv-python (install via:
  pip install numpy opencv-python
in the Python environment Webots is configured to use -- check
Tools > Preferences > General > Python command in Webots.)
"""

from controller import Robot
import numpy as np
import cv2
import csv
import os

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

# 8 IR distance sensors, named ps0..ps7 on the e-puck
NUM_SENSORS = 8
distance_sensors = []
for i in range(NUM_SENSORS):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(TIME_STEP)
    distance_sensors.append(sensor)

# Camera
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)
cam_width = camera.getWidth()
cam_height = camera.getHeight()

# Wheel motors (left/right) - used later for control, kept simple here
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Output log file
os.makedirs("../../data", exist_ok=True)
log_path = "../../data/perception_log.csv"
log_file = open(log_path, mode="w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(
    ["timestep"] + [f"ps{i}" for i in range(NUM_SENSORS)]
    + ["num_obstacles_detected", "obstacle_centroids_xy"]
)


def detect_obstacles(image_bgr):
    """
    Simple OpenCV pipeline to detect the coloured obstacle blocks
    placed in the world (red / green / blue / yellow boxes).

    Steps (documented for report justification):
      1. Convert BGR -> HSV: colour thresholding is far more robust
         to lighting changes in HSV than in raw RGB/BGR space.
      2. Threshold for saturated, non-background colours (background
         is grey/white floor + walls).
      3. Morphological close to remove speckle noise from render
         aliasing.
      4. Contour detection -> filter by minimum area to reject noise.
      5. Return centroid (pixel coords) of each detected obstacle.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Background (floor/walls) is low-saturation grey/white.
    # Obstacles are the saturated PBRAppearance colours we defined
    # in the world file, so thresholding on saturation channel
    # isolates them cleanly regardless of exact hue.
    lower = np.array([0, 80, 40])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centroids = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:  # reject tiny noise blobs
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))

    return centroids


# ---------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------
step_count = 0
MAX_STEPS = 300  # ~ a few seconds of logging; adjust as needed

while robot.step(TIME_STEP) != -1 and step_count < MAX_STEPS:
    # --- Read distance sensors ---
    ps_values = [s.getValue() for s in distance_sensors]

    # --- Read camera image ---
    image = camera.getImage()  # raw byte buffer BGRA
    img_array = np.frombuffer(image, np.uint8).reshape(
        (cam_height, cam_width, 4)
    )
    img_bgr = img_array[:, :, :3]  # drop alpha channel

    centroids = detect_obstacles(img_bgr)

    # --- Log synchronised record ---
    log_writer.writerow(
        [step_count] + ps_values + [len(centroids), centroids]
    )

    # --- Simple placeholder behaviour for Day 1: stay stationary
    # while we validate sensor + vision pipeline. Motion/PID control
    # comes in Day 4-5. ---
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    step_count += 1

log_file.close()
print(f"Perception logging complete. {step_count} steps written to {log_path}")