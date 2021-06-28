"""Auto-aims catapult using 60 degree arc laptop camera."""
import cv2
import serial
import numpy as np
import matplotlib.pyplot as plt
import math
import utils

from dataclasses import dataclass
from detectron2.config import get_cfg, CfgNode
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


def configure_detectron(mode: str = 'cpu',
                        threshold: float = .8,
                        config_location: str = 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml',
                        weights_location: str = 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml'
                        ) -> CfgNode:
    """Configures detectron configuration object and returns it's interface."""
    cfg = get_cfg()
    cfg.MODEL.DEVICE = mode
    cfg.merge_from_file(model_zoo.get_config_file(config_location))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_location)
    return cfg

def mask_dimensions(image: cv2.imread, cfg: CfgNode, mode: str = 'cpu'
                    ) -> list:
    """Returns image recognition mask dimensions for red cup.

    Configures detectron config object with pretrained model, creates
    api for model detection, and returns formatted rectangular coordinate
    list with pixel dimensions in format [x1, y1, x2, y2].

    :returns: Rectangular dimensions for detection mask
    :rtype: list
    """
    model = DefaultPredictor(cfg)
    prediction = model(image)
    # oh god no
    # print(prediction['instances'].pred_classes[41])
    # # return prediction['instances'].pred_boxes[41].tensor.cpu().numpy()[0].astype(np.int32)
    # return np.array(prediction['instances'].pred_boxes[])

    print(prediction['instances'].pred_classes)
    print(prediction['instances'].pred_classes == 41)


def mask_aspect_ratio(image: cv2.imread, mask: list) -> AspectRatio:
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
    x_ratio = ((image.shape[1]/2) - (mask[2] - mask[0]) / image.shape[1]/2)
    y_ratio = ((image.shape[0]/2) - (mask[3] - mask[1]) / image.shape[0]/2)

    return AspectRatio(x_ratio, y_ratio)


def launch_angle(distance_camera_to_structure: int, aspect_ratios: AspectRatio,
                 in_per_ppx: int = 5, laptop_arc: int = 30):
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
    radius = (aspect_ratio.x_ratio * in_per_ppx) / (math.pi/3)
    x_angle = aspect_ratios.x_ratio * laptop_arc #degrees

    # Law of cosines to calculate missing side length
    distance_structure_to_structure = utils.cosines_law(b=radius,
                                    c=distance_camera_to_structure, A=x_angle)

    theta = utils.sines_law(a=distance_structure_to_structure,
                            b=distance_camera_to_structure, B=x_angle)

    return 90 - theta, aspect_ratios.y_ratio * 30


def show_image(image: cv2.imread):
    cv2.imshow("cup", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread('cup.jpg')
    show_image(image)
    print(mask_dimensions(image=image, cfg=configure_detectron(), mode='cpu'))
