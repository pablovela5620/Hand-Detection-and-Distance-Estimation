from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
from imutils.video import VideoStream, FPS

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    # Detection confidence threshold to draw bounding box
    score_thresh = 0.60

    # Get stream from webcam and set parameters)
    vs = VideoStream().start()

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0
    im_fps = FPS().start()

    im_height, im_width = (vs.read().shape[0], vs.read().shape[1])
    # max number of hands we want to detect/track
    num_hands_detect = 2

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np = vs.read()

        # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Run image through tensorflow graph
        boxes, scores, classes = detector_utils.detect_objects(
            image_np, detection_graph, sess)

        # Draw bounding boxeses and text
        detector_utils.draw_box_on_image(
            num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time
        im_fps.update()

        # Display FPS on frame
        detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), image_np)
        cv2.imshow('Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            vs.stop()
            break
    im_fps.stop()
    print("Average FPS: ", im_fps.fps())
