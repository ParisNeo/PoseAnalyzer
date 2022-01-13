"""=== hello_poses =>

    Author : Saifeddine ALOUI
    Description :
        A code to test PoseAnalyzer: Extract poses landmarks from a realtime video input
<================"""
from PoseAnalyzer import PoseAnalyzer, Pose
from PoseAnalyzer.helpers.geometry.orientation import orientation2Euler
import numpy as np
import cv2
from pathlib import Path

from HandsAnalyzer.helpers.geometry.euclidian import get_alignment_coefficient

# open camera
cap = cv2.VideoCapture(0)

# Build a window
cv2.namedWindow('Hello poses', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hello poses', (640,480))

# Build face analyzer while specifying that we want to extract just a single face
ha = PoseAnalyzer()
y,p,r=0,0,0
# Main Loop
while cap.isOpened():
    # Read image
    success, image = cap.read()
    # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image to extract poses and draw the lines
    ha.process(image)
    # If there are some poses then process
    if ha.nb_poses>0:
        pose = ha.pose
        # Draw the landmarks
        pose.draw_landmarks(image, thickness=3)
        left_arm_lm = pose.get_landmarks_pos(pose.left_arm_indices)
        right_arm_lm = pose.get_landmarks_pos(pose.right_arm_indices)
        if left_arm_lm[2,1]<left_arm_lm[1,1] and right_arm_lm[2,1]<right_arm_lm[1,1]:
            # Draw a bounding box
            pose.annotate(image,text=f"Left and right arms raised")
        elif left_arm_lm[2,1]<left_arm_lm[1,1]:
            # Draw a bounding box
            pose.annotate(image,text=f"Left arm raised")
        elif right_arm_lm[2,1]<right_arm_lm[1,1]:
            # Draw a bounding box
            pose.annotate(image,text=f"Right arm raised")

    # Show the image
    try:
        cv2.imshow('Hello poses', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as ex:
        print(ex)
    
    # Wait for key stroke for 5 ms
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break
    if wk & 0xFF == 115: # If s is pressed then take a snapshot
        sc_dir = Path(__file__).parent/"screenshots"
        if not sc_dir.exists():
            sc_dir.mkdir(exist_ok=True, parents=True)
        i = 1
        file = sc_dir /f"sc_{i}.jpg"
        while file.exists():
            i+=1
            file = sc_dir /f"sc_{i}.jpg"
        cv2.imwrite(str(file),cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        print(pose.get_landmarks_pos(pose.palm_indices))
        print("Shot")



