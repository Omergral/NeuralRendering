import cv2
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


class RGBDetection:

    def __init__(self, image, output_path: str = None, visualize: bool = False):
        if type(image) == str:
            self.image = cv2.imread(image)
        else:
            self.image = image[:, :, :-1]
        self.green_lower, self.green_upper = ((0, 100, 0), (100, 255, 100))
        self.red_lower, self.red_upper = ((0, 0, 100), (100, 100, 255))
        self.blue_lower, self.blue_upper = ((100, 0, 0), (255, 100, 100))
        self.visualize = visualize
        self.out_path = Path(output_path) if output_path else None

    def get_coordinates(self):
        mask_cone = cv2.inRange(self.image, self.red_lower, self.red_upper)
        mask_cube_blue = cv2.inRange(self.image, self.blue_lower, self.blue_upper)
        mask_cube_green = cv2.inRange(self.image, self.green_lower, self.green_upper)

        x_cube_green, y_cube_green, w_cube_green, h_cube_green = cv2.boundingRect(mask_cube_green)
        x_cube_blue, y_cube_blue, w_cube_blue, h_cube_blue = cv2.boundingRect(mask_cube_blue)
        x_cone, y_cone, w_cone, h_cone = cv2.boundingRect(mask_cone)

        green_cube_coord = (x_cube_green, y_cube_green, w_cube_green, h_cube_green)
        blue_cube_coord = (x_cube_blue, y_cube_blue, w_cube_blue, h_cube_blue)
        red_cone_coord = (x_cone, y_cone, w_cone, h_cone)

        return green_cube_coord, blue_cube_coord, red_cone_coord

    def draw_rects(self):

        green_cube_coord, blue_cube_coord, red_cone_coord = self.get_coordinates()
        green_cube_coord_start, green_cube_coord_end = ((green_cube_coord[0], green_cube_coord[1]),
                                                        (green_cube_coord[0] + green_cube_coord[2],
                                                         green_cube_coord[1] + green_cube_coord[3]))
        blue_cube_coord_start, blue_cube_coord_end = ((blue_cube_coord[0], blue_cube_coord[1]),
                                                      (blue_cube_coord[0] + blue_cube_coord[2],
                                                       blue_cube_coord[1] + blue_cube_coord[3]))
        cone_coord_start, cone_coord_end = ((red_cone_coord[0], red_cone_coord[1]),
                                            (red_cone_coord[0] + red_cone_coord[2],
                                             red_cone_coord[1] + red_cone_coord[3]))

        image_copy = self.image.copy()

        cv2.rectangle(image_copy, cone_coord_start, cone_coord_end, (0, 0, 255))
        cv2.rectangle(image_copy, blue_cube_coord_start, blue_cube_coord_end, (255, 0, 0))
        cv2.rectangle(image_copy, green_cube_coord_start, green_cube_coord_end, (0, 255, 0))

        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        if self.out_path:
            cv2.imwrite(str(self.out_path / 'rects.png'), image_copy)


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

    green_cube_coord, blue_cube_coord, red_cone_coord = object_detector.get_coordinates()

    if args.visualize:

        object_detector.draw_rects()