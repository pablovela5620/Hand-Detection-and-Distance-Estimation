from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    # Detection confidence threshold to draw bounding box
    score_thresh = 0.60

    # Get stream from webcam and set parameters
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()

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

        # Display FPS on frame
        detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), image_np)
        cv2.imshow('Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
