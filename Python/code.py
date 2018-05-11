import argparse
import os.path

import cv2
import imutils
import numpy as np


def main(image_path, debug):
    """ Main function of the Test """
     # Load image
    image = load_image(image_path)

    # Resize the image
    image = imutils.resize(image, width=1600)

    # Convert to gray and apply bilateral filter to smooth the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral_filter = cv2.bilateralFilter(gray, 25, 10, 80)

    # Binarize the image and find the countours of the shape
    binary = cv2.threshold(bilateral_filter, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    # Create a blank image and draw countours find inside
    height, width, _ = image.shape
    blank_image = np.zeros((height, width), np.uint8)
    for contour in contours:
        contour = remove_pointless(contour)
        cv2.drawContours(blank_image, [contour], -1, (255, 255, 255), 1)

    # Invert the image and convert it to RGB
    inv_blank_image = ~blank_image
    color = cv2.cvtColor(inv_blank_image, cv2.COLOR_GRAY2BGR)

    # Add rectangle if it's not cross a contour 
    p1 = (0,0)
    p2 = (450,450)

    if check_number_contour(len(contours), binary, p1, p2):
        cv2.rectangle(color, p1, p2, (0,255,0), -1)

    if debug:
        cv2.imshow("New Image After dilation", color)
        cv2.waitKey()

        # Wait Esc key to stop
        while True:
            key = cv2.waitKey()
            if key == 27:
                cv2.destroyAllWindows()
                exit()

def remove_pointless(contour):
    new_contour = []
    i = 1

    while i < len(contour):
        if abs(contour[i][0][0] - contour[i-1][0][0]) == 1 and abs(contour[i][0][1] - contour[i-1][0][1]) == 1:
            tmp = []
            if contour[i-1][0][0] < contour[i][0][0]: 
                if contour[i-1][0][1] > contour[i][0][1]:
                    tmp.append(contour[i-1][0][0])
                    tmp.append(contour[i][0][1])
                else:
                    tmp.append(contour[i][0][0])
                    tmp.append(contour[i-1][0][1])
            else:
                if contour[i-1][0][1] > contour[i][0][1]:
                    tmp.append(contour[i][0][0])
                    tmp.append(contour[i-1][0][1])
                else:
                    tmp.append(contour[i-1][0][0])
                    tmp.append(contour[i][0][1])

            new_contour.append([tmp])
            i += 2
        else:
            new_contour.append(contour[i-1])
            if(i + 1 == len(contour)):
                new_contour.append(contour[i])

            i += 1
    
    return np.array(new_contour)

def check_number_contour(actual_number_contours, binary, p1, p2):
    tmp_binary = binary.copy() 
    cv2.rectangle(tmp_binary, p1, p2, (255,255,255), -1)
    tmp_contours = cv2.findContours(tmp_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    return len(tmp_contours) > actual_number_contours

def load_image(image_path):
    """ Read the image and return a 3-dimensional matrix """

    if not os.path.isfile(image_path):
        print("You have passed an invalid image")
        exit()

    return cv2.imread(image_path)

def parse_arguments():
    """ Setup up a parser to read command line arguments"""

    parser = argparse.ArgumentParser(
        description='Test')
    parser.add_argument('-i', '--image',
                        action='store',
                        dest='image_path',
                        required=True,
                        help='path of the image to read')
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        dest='debug',
                        help='debug flag to display all transition images')
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s 0.0.1')
    return parser.parse_args()

if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments.image_path, arguments.debug)
