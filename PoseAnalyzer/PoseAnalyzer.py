# -*- coding: utf-8 -*-
"""=== Pose Analyzer =>
    Module : PoseAnalyzer
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Main module. PoseAnalyzer analyzes an image and extract poses by creating instances of Pose in its attribute Faces
<================"""

from typing import NamedTuple, Tuple
import numpy as np
import mediapipe as mp
import cv2
import math
import time
from PIL import Image
from scipy.signal import butter, filtfilt

from .Pose import Pose

class PoseAnalyzer():
    """A class that analyzes the facial components
    """

    def __init__(self,  image_shape: tuple = (640, 480),  static_image_mode:bool=True, model_complexity:int=2, min_detection_confidence:float=0.5):
        """Creates an instance of the PoseAnalyzer object

        Args:
            max_nb_poses (int,optional) : The maximum number of poses to be detected by the mediapipe library
            image_shape (tuple, optional): The shape of the image to be processed. Defaults to (480, 640).
            model_complexity (int,optional) : The complexity level of the model to be used ()
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.fmd = mp.solutions.pose.Pose(static_image_mode=static_image_mode, model_complexity=model_complexity, min_detection_confidence= min_detection_confidence)


        self.pose = Pose(image_shape=image_shape)
        self.image_shape = image_shape
        self.image = None
        self.results = None
        self.found_poses = False
        self.found_poses = False
        self.nb_poses = 0        

    @property
    def image_size(self)->tuple:
        """A property to image size

        Returns:
            tuple: The image size
        """
        return self.image_shape

    @image_size.setter
    def image_size(self,new_shape:tuple):
        self.image_shape=new_shape
        for pose in self.poses:
            pose.image_shape=new_shape

    def process(self, image: np.ndarray) -> NamedTuple:
        """Processes an image and extracts the poses

        Args:
            image (np.ndarray): The image to extract poses from

        Returns:
            NamedTuple: The result of extracting the image
        """
        # Process the image
        results = self.fmd.process(image)

        # Keep a local reference to the image
        self.image = image

        # If poses found
        if results.pose_landmarks is not None:
            self.found_poses = True
            self.nb_poses = 1
        else:
            self.found_poses = False
            self.nb_poses = 0
            return
    
        # Update poses
        self.pose.update(results.pose_landmarks)

        self.results = results
    @staticmethod
    def from_image(file_name:str, max_nb_poses:int=1, image_shape:tuple=(640, 480)):
        """Opens an image and extracts a pose from it

        Args:
            file_name (str)                 : The name of the image file containing one or multiple poses
            max_nb_poses (int, optional)    : The maximum number of poses to extract. Defaults to 1
            image_shape (tuple, optional)   : The image shape. Defaults to (640, 480)

        Returns:
            An instance of PoseAnalyzer: A pose analyzer containing all processed poses out of the image. Ile image can be found at fa.image
        """
        fa = PoseAnalyzer(max_nb_poses=max_nb_poses)
        image = Image.open(file_name)
        image = image.resize(image_shape)
        npImage = np.array(image)[...,:3]
        fa.process(npImage)
        return fa
