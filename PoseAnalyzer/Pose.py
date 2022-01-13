# -*- coding: utf-8 -*-
"""=== Pose Analyzer =>
    Module : Pose
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Pose data holder (landmarks, posture ...). Allows the extraction of multiple facial features out of landmarks.
<================"""


import re
from typing import NamedTuple, Tuple
import numpy as np
import mediapipe as mp
import cv2
from numpy import linalg
from numpy.lib.type_check import imag
from scipy.signal import butter, filtfilt
import math
import time
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R


from .helpers.geometry.euclidian import buildCameraMatrix, get_plane_infos, get_z_line_equation, get_plane_line_intersection, get_alignment_coefficient
from .helpers.geometry.orientation import rotateLandmarks

# Get an instance of drawing specs to be used for drawing masks on hands
DrawingSpec =  mp.solutions.drawing_utils.DrawingSpec
class Pose():    
    """Pose is the class that provides operations on pose landmarks.
    It is extracted by the pose analyzer and could then be used for multiple pose features extraction purposes
    """
    # Body joints
    nose_index=0
    left_eye_inner_index=1
    left_eye_index=2
    left_eye_outer_index=3
    right_eye_inner_index=4
    right_eye_index=5
    right_eye_outer_index=6
    left_ear_index=7
    right_ear_index=8
    mouth_left_index=9
    mouth_right_index=10
    left_shoulder_index=11
    right_shoulder_index=12
    left_elbow_index=13
    right_elbow_index=14
    left_wrist_index=15
    right_wrist_index=16
    left_pinky_index=17
    right_pinky_index=18
    left_index_index=19
    right_index_index=20
    left_thumb_index=21
    right_thumb_index=22

    left_hip_index=23
    right_hip_index=24

    left_knee_index=25
    right_knee_index=26

    left_ankle_index=27
    right_ankle_index=28

    left_heel_index=29
    right_heel_index=30

    left_foot_index_index=31
    right_foot_index_index=32

    # Body parts
    left_eye_indices    =[left_eye_inner_index, left_eye_index, left_eye_outer_index]
    right_eye_indices   =[right_eye_inner_index, right_eye_index, right_eye_outer_index]

    mouth_indices       =[mouth_left_index, mouth_right_index]

    main_body_indices   =[left_shoulder_index, right_shoulder_index, right_hip_index, left_hip_index]

    left_arm_indices    =[left_shoulder_index, left_elbow_index, left_wrist_index]
    right_arm_indices   =[right_shoulder_index, right_elbow_index, right_wrist_index]

    left_hand_indices    =[left_wrist_index, left_index_index, left_pinky_index]
    right_hand_indices    =[right_wrist_index, right_index_index, right_pinky_index]

    left_leg_indices    =[left_hip_index, left_knee_index, left_ankle_index]
    right_leg_indices   =[right_hip_index, right_knee_index, right_ankle_index]

    left_foot_indices    =[left_ankle_index, left_heel_index, left_foot_index_index]
    right_foot_indices    =[right_ankle_index, right_heel_index, right_foot_index_index]


    def __init__(self, landmarks:NamedTuple = None, image_shape: tuple = (640, 480)):
        """Creates an instance of Pose

        Args:
            is_left (bool): if true, this is a left pose else this is a right pose.
            landmarks (NamedTuple, optional): Landmarks object extracted by mediapipe tools
            image_shape (tuple, optional): The width and height of the image used to extract the pose. Required to get landmarks in the right pixel size (useful for pose copying and image operations). Defaults to (480, 640).
        """
        self.image_shape = image_shape

        if type(landmarks)==np.ndarray:
            self.npLandmarks=landmarks
        else:
            self.update(landmarks)


        # Initialize pose information
        self.pos = None
        self.ori = None

        self.mp_drawing = mp.solutions.drawing_utils



    @property
    def ready(self)->bool:
        """Returns if the pose has landmarks or not

        Returns:
            bool: True if the pose has landmarks
        """
        return self.landmarks is not None

    def update(self, landmarks:NamedTuple)->None:
        """Updates the landmarks of the pose

        Args:
            is_left (bool): if true, this is a left pose else this is a right pose.
            landmarks (NamedTuple): The new landmarks
        """
        if landmarks is not None:
            self.landmarks = landmarks
            self.npLandmarks = np.array([[lm.x * self.image_shape[0], lm.y * self.image_shape[1], lm.z * self.image_shape[0]] for lm in landmarks.landmark])
        else:
            self.landmarks = None
            self.npLandmarks = np.array([])


    def get_landmark_pos(self, index) -> Tuple:
        """Recovers the position of a landmark from a results array

        Args:
            index (int): Index of the landmark to recover

        Returns:
            Tuple: Landmark 3D position in image space
        """

        # Assertion to verify that the pose object is ready
        assert self.ready, "Pose object is not ready. There are no landmarks extracted."

        lm = self.npLandmarks[index, ...]
        return np.array([lm[0], lm[1], lm[2]])



    def get_landmarks_pos(self, indices: list) -> np.ndarray:
        """Recovers the position of a landmark from a results array

        Args:
            indices (list): List of indices of landmarks to extract

        Returns:
            np.ndarray: A nX3 array where n is the number of landmarks to be extracted and 3 are the 3 cartesian coordinates
        """

        # Assertion to verify that the pose object is ready
        assert self.ready, "Pose object is not ready. There are no landmarks extracted."

        return self.npLandmarks[indices,...]


    def draw_landmark_by_index(self, image: np.ndarray, index: int, color: tuple = (255, 0, 0), radius: int = 5, thickness:int=1) -> np.ndarray:
        """Draw a landmark on an image from landmark index

        Args:
            image (np.ndarray): Image to draw the landmark on
            index (int): Index of the landmark
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.

        Returns:
            np.ndarray: Output image
        """
        pos = self.npLandmarks[index,:]
        return cv2.circle(
            image,(int(pos[0]), int(pos[1])), radius, color, thickness
        )

    def draw_links(self, image: np.ndarray, landmarks, color: tuple = (255, 0, 0), thickness: int = 1, close:bool=False) -> np.ndarray:
        if close:
            for i in range(landmarks.shape[0]):
                image = cv2.line(image, (int(landmarks[i,0]), int(landmarks[i,1])),(int(landmarks[(i+1)%landmarks.shape[0],0]), int(landmarks[(i+1)%landmarks.shape[0],1])),color, thickness)
        else:
            for i in range(landmarks.shape[0]-1):
                image = cv2.line(image, (int(landmarks[i,0]), int(landmarks[i,1])),(int(landmarks[(i+1),0]), int(landmarks[(i+1),1])),color, thickness)

    def draw_landmarks(
                self, 
                image: np.ndarray, 
                landmarks: np.ndarray=None, 
                radius:int=1, 
                color: tuple = (255, 0, 0), 
                thickness: int = 1, 
                link=True
                ) -> np.ndarray:
        """Draw a list of landmarks on an image

        Args:
            image (np.ndarray): Image to draw the contour on
            landmarks (np.ndarray): a nX3 ndarray containing the positions of the landmarks. Defaults to None (use all landmarks).
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.


        Returns:
            np.ndarray: The image with the contour drawn on it
        """
        if landmarks is None:
            landmarks = self.npLandmarks
            # Here we draw our landmarks
            lm_l=landmarks.shape[0]
            for i in range(lm_l):
                image = cv2.circle(image, (int(landmarks[i,0]), int(landmarks[i,1])), radius,color, thickness)
            if link:
                # Draw body parts
                self.draw_links(image, self.npLandmarks[self.left_eye_indices,...], color, thickness)
                self.draw_links(image, self.npLandmarks[self.right_eye_indices,...], color, thickness)
                self.draw_links(image, self.npLandmarks[self.mouth_indices,...], color, thickness)
                self.draw_links(image, self.npLandmarks[self.main_body_indices,...], color, thickness, close=True)
                self.draw_links(image, self.npLandmarks[self.left_arm_indices,...], color, thickness)
                self.draw_links(image, self.npLandmarks[self.right_arm_indices,...], color, thickness)
                self.draw_links(image, self.npLandmarks[self.left_hand_indices,...], color, thickness, close=True)
                self.draw_links(image, self.npLandmarks[self.right_hand_indices,...], color, thickness, close=True)
                
                self.draw_links(image, self.npLandmarks[self.left_leg_indices,...], color, thickness)
                self.draw_links(image, self.npLandmarks[self.right_leg_indices,...], color, thickness)
                

            return image


        lm_l=landmarks.shape[0]
        for i in range(lm_l):
            image = cv2.circle(image, (int(landmarks[i,0]), int(landmarks[i,1])), radius,color, thickness)
            if link:
                if i not in [4,8,12,16,20]:
                    image = cv2.line(image, (int(landmarks[i,0]), int(landmarks[i,1])),(int(landmarks[(i+1)%lm_l,0]), int(landmarks[(i+1)%lm_l,1])),color, thickness)
        return image

    def draw_landmark(self, image: np.ndarray, pos: tuple, color: tuple = (255, 0, 0), radius: int = 5, thickness:int=1) -> np.ndarray:
        """Draw a landmark on an image

        Args:
            image (np.ndarray): Image to draw the landmark on
            pos (tuple): Position of the landmark
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.

        Returns:
            np.ndarray: Output image
        """
        return cv2.circle(
            image,(int(pos[0]), int(pos[1])), radius, color, thickness
        )

    def draw_contour(self, image: np.ndarray, contour: np.ndarray, color: tuple = (255, 0, 0), thickness: int = 1, isClosed:bool = True) -> np.ndarray:
        """Draw a contour on an image

        Args:
            image (np.ndarray): Image to draw the contour on
            contour (np.ndarray): a nX3 ndarray containing the positions of the landmarks
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.
            isClosed (bool, optional): If True, the contour will be closed, otherwize it will be kept open. Defaults to True 


        Returns:
            np.ndarray: The image with the contour drawn on it
        """

        pts = np.array([[int(p[0]), int(p[1])] for p in contour.tolist()]).reshape((-1, 1, 2))
        return cv2.polylines(image, [pts], isClosed, color, thickness)

    def get_hand_posture(self, camera_matrix:np.ndarray = None, dist_coeffs:np.ndarray=np.zeros((4,1)))->tuple:
        """Gets the posture of the head (position in cartesian space and Euler angles)
        Args:
            camera_matrix (int, optional)       : The camera matrix built using buildCameraMatrix Helper function. Defaults to a perfect camera matrix 
            dist_coeffs (np.ndarray, optional)) : The distortion coefficients of the camera
        Returns:
            tuple: (position, orientation) the orientation is either in compact rodriguez format (angle * u where u is the rotation unit 3d vector representing the rotation axis). Feel free to use the helper functions to convert to angles or matrix
        """

        # Assertion to verify that the pose object is ready
        assert self.ready, "Pose object is not ready. There are no landmarks extracted."

        if camera_matrix is None:
            camera_matrix= buildCameraMatrix()

        # Use opencv's PnPsolver to solve the rotation problem

        face_2d_positions = self.npLandmarks[self.palm_indices,:2]
        if self.is_left:
                (success, face_ori, face_pos, _) = cv2.solvePnPRansac(
                                                            self.left_palm_reference_positions.astype(np.float),
                                                            face_2d_positions.astype(np.float), 
                                                            camera_matrix, 
                                                            dist_coeffs)#,
                                                            #flags=cv2.SOLVEPNP_ITERATIVE)
        else:
                (success, face_ori, face_pos, _) = cv2.solvePnPRansac(
                                                            self.left_palm_reference_positions.astype(np.float),
                                                            face_2d_positions.astype(np.float), 
                                                            camera_matrix, 
                                                            dist_coeffs)#,
                                                            #flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None, None

        # save posture
        self.pos = face_pos
        self.ori = face_ori

        return face_pos, face_ori


    def rect_contains(self, rect:tuple, point:tuple)->bool:
        """Tells whether a point is inside a rectangular region

        Args:
            rect (tuple): The rectangle coordiantes (topleft , bottomright)
            point (tuple): The point position (x,y)

        Returns:
            bool: True if the point is inside the rectangular region
        """
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True


    def getBodyBox(self, image:np.ndarray, landmark_indices:list=None, margins=(0,0,0,0))->np.ndarray:
        """Gets an image of the pose extracted from the original image (simple box extraction which will extract some of the background)

        Args:
            image (np.ndarray): Image to extract the pose from
            src_triangles (list): The delaulay triangles indices (look at triangulate)
            landmark_indices (list, optional): The list of landmarks to be used (the same list used for the triangulate method that allowed the extraction of the triangles). Defaults to None.

        Returns:
            np.ndarray: Pose drawn on a black background (the size of the image is equal of that of the pose in the original image)
        """

        # Assertion to verify that the pose object is ready
        assert self.ready, "Pose object is not ready. There are no landmarks extracted."

        if landmark_indices is None:
            landmarks = self.npLandmarks[:, :2]
        else:
            landmarks = self.npLandmarks[landmark_indices, :2]
        p1 = landmarks.min(axis=0)-np.array(margins[0:2])
        p2 = landmarks.max(axis=0)+np.array(margins[2:4])
        return image[int(p1[1]):int(p2[1]),int(p1[0]):int(p2[0])]

    def annotate(self, image:np.ndarray, text:str, color:tuple=(255,0,0), thickness:int=1, pos:np.ndarray=None):
        """Writes an annotation on top of the object

        Args:
            image (np.ndarray): The image on which we will draw the bounding box
            text (str) : the text to be displayed
            color (tuple, optional): The color of the bounding box. Defaults to (255,0,0).
            thickness (int, optional): The line thickness. Defaults to 1.
        """
        if pos is None:
            pos = self.npLandmarks.min(axis=0)
        cv2.putText(image, text, (int(pos[0]),int(pos[1]-20)),cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

    def draw_bounding_box(self, image:np.ndarray, color:tuple=(255,0,0), thickness:int=1, text=None):
        """Draws a bounding box around the pose

        Args:
            image (np.ndarray): The image on which we will draw the bounding box
            color (tuple, optional): The color of the bounding box. Defaults to (255,0,0).
            thickness (int, optional): The line thickness. Defaults to 1.
        """
        pt1 = self.npLandmarks.min(axis=0)
        pt2 = self.npLandmarks.max(axis=0)
        cv2.rectangle(image, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), color, thickness)
        if text is not None:
            cv2.putText(image, text, (int(pt1[0]),int(pt1[1]-20)),cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

