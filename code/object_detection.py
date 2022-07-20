import cv2
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


class RGBDetection:

    def __init__(self, image, output_path: str = None, visualize: bool = False):
        self.image = image[:, :, :3]
        self.green_bound = (0, 255, 0)
        self.red_bound = (255, 0, 0)
        self.blue_bound = (0, 0, 255)
        self.purple_bound = (100, 0, 100)
        self.tuqruoise_bound = (0, 255, 255)
        self.yellow_bound = (255, 255, 0)
        self.orange_bound = (255, 153, 18)
        self.visualize = visualize
        self.out_path = Path(output_path) if output_path else None
        self.colors = {'green': self.green_bound, 'red': self.red_bound, 'blue': self.blue_bound,
                       'purple': self.purple_bound, 'yellow': self.yellow_bound, 'orange': self.orange_bound,
                       'turquoise': self.tuqruoise_bound}

    def get_coordinates(self):
        masks = {}
        bound_coordinates = {}

        for color, bound in self.colors.items():
            masks[color] = cv2.inRange(self.image, bound, bound)

        for color, col_range in masks.items():
            bound_coordinates[color] = cv2.boundingRect(col_range)

        return bound_coordinates

    def draw_rects(self):
        rect_coordinates = {}
        color_coords = self.get_coordinates()

        for color, bound_coordinates in color_coords.items():
            x, y, w, h = bound_coordinates
            rect_coordinates[color] = ((x, y), (x + w, y + h))

        image_copy = self.image.copy()

        for color, rect_coords in rect_coordinates.items():
            cv2.rectangle(image_copy, rect_coords[0], rect_coords[1], (0, 0, 0))

        plt.imshow(image_copy)
        plt.axis('off')

        if self.out_path:
            cv2.imwrite(str(self.out_path / 'rects.png'), cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('image', type=str, help='path to image')

    parser.add_argument('-o', '--output_path', type=str, required=False,
                        help="path to output dir if 'visualize' set to 'True' and you want to save the image with"
                             "the rectangles")

    parser.add_argument('-v', '--visualize', type=bool, required=False, default=False,
                        help="set to 'True' if you want to visualize the detection of the objects")

    args = parser.parse_args()

    object_detector = RGBDetection(image=args.image, output_path=args.output_path, visualize=args.visualize)

    color_coordinates = object_detector.get_coordinates()

    if args.visualize:

        object_detector.draw_rects()
