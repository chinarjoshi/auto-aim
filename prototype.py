import cv2
import serial
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
import utils

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
# ----------------------------------------------------------

@dataclass
class AspectRatio:
    """Represents an x and y aspect ratio."""
    x_ratio: int
    y_ratio: int


@dataclass
class CupModel:
    """Represents image of plastic cup."""
    image: cv2.imread

    def mask_dimensions(self) -> list:
        """Returns image recognition mask dimensions for red cup.

        Configures detectron config object with pretrained model, creates
        api for model detection, and returns formatted rectangular coordinate
        list with pixel dimensions in format [x1, y1, x2, y2].

        :returns: Rectangular dimensions for detection mask
        :rtype: list
        """
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

        model = DefaultPredictor(cfg)
        prediction = model(self.image)

        # oh god no
        return prediction['instances'].pred_boxes.tensor.cpu().numpy()[0].astype(np.int32)


    def mask_aspect_ratio(self) -> AspectRatio:
        """Uses trignometric relationships to calculate angular displacement.

        The whole system is represented as the following model:

         [fort2]
            |  \
            |   \
     (10ft) |    \
            |     \
            |      \
            |    (θ)\
         [fort1] [laptop]
              (1ft)

        Since the camera will always be facing directly at the opposing structure,
        the one value that determines translational launch angle is the midpoint
        of the rectangular mask represented as a pixel ratio.
        """
        mask = self.mask_dimensions()
        x_ratio = ((self.image.shape[1]/2) - (mask[2] - mask[0]) / self.image.shape[1]/2)
        y_ratio = ((self.image.shape[0]/2) - (mask[3] - mask[1]) / self.image.shape[0]/2)

        return AspectRatio(x_ratio, y_ratio)


    def launch_angle(self):
        """Using the mask's aspect ratio, returns launch angle.

        Returns the angle at which the catapult should rotate from 0 degrees angular
        displacement, from a range of [-15, 15](degrees). The data will then be
        serialized and sent to arduino.
        The camera on my computer goes approximately 30 degrees in either direction,
        thus the fraction of mask-midpoint/(dimension/2) times 30 should return the

        Cause the thing is, for it to be considered autonomous in the competition,
        the laptop must be a constant distance and angle away from the opposing
        structure. ***Thus, so long as the laptop is that constant distance and angle
        away, it really doesn't matter where the opposing structure is. The angle is
        calculated from the distance from image recognition in the camera, and then the
        laptop will be facing a certain amount of degrees from the structure.

        ***PROBLEM***
        You also need the adjusted distance between the structures because the laptop
        and structure face two different angles.
        **SOLUTION**
        It should be a simple arclength problem. The dimensions we see on screen is
        a flattened arc, and thus to find the radius we simply need to divide the aspect
        ratio of the object by the aspect ratio of the whole camera, being pi/3 (60 deg).

        The laptop angular displacement is calculated from the formula:
          arctan({distance of the two structures apart}/{laptop distance to structure})

        Assume the inital angular displacement of the laptop is 78.69 degrees.

        We have a SAS scalene triangle from the resultant measurements, thus we can
        use law of sines to find the distance between the two structures, and then
        plug that into the law of cosines to find delta theta.

        Although a linear regression model can be trained using annotated pictures with
        distances, this method follows a purely mathematical approach.
        """
        # TODO Some constant inch per pixel
        constant = 5
        aspect_ratios = self.mask_aspect_ratio()

        distance_camera_to_structure = 2 #ft
        radius = (aspect_ratio.x_ratio * constant) / (math.pi/3)
        x_angle = aspect_ratios.x_ratio * 30 #degrees

        # Law of cosines to calculate missing side length
        distance_structure_to_structure = utils.cosines_law(b=radius,
                                        c=distance_camera_to_structure, A=x_angle)

        theta = utils.sines_law(a=distance_structure_to_structure,
                                b=distance_camera_to_structure, B=x_angle)

        return 90 - theta, aspect_ratios.y_ratio * 30


    def send_to_arduino(device: str = 'dev/ttyACM1', baud: int = 9600):
        ser = serial.Serial(device, baud)
        ser.write(self.launch_angle())


    def show_image(self):
        cv2.imshow("cup", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cup = CupModel(cv2.imread('cup.jpg'))
