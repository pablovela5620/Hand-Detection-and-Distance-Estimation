import cv2
import datetime
import argparse
import imutils
from imutils.video import VideoStream

from utils import detector_utils as detector_utils

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.60

    # Get stream from webcam and set parameters)
    vs = VideoStream().start()

    # max number of hands we want to detect/track
    num_hands_detect = 1

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)

    try:
        while True:
            # Read Frame and process
            frame = vs.read()
            frame = cv2.resize(frame, (320, 240))

            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # Draw bounding boxeses and text
            detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break

        print("Average FPS: ", str("{0:.2f}".format(fps)))

    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))
