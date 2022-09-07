"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect human face in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

To find more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2
import pandas as pd
import datetime

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


if __name__ == '__main__':
    # Before estimation started, there are some startup works to do.

    # 1. Setup the video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    # Get the frame size. This will be used by the pose estimator.
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 2. Introduce a pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(height, width))

    # 3. Introduce a mark detector to detect landmarks.
    mark_detector = MarkDetector()

    # 4. Measure the performance with a tick meter.
    tm = cv2.TickMeter()

    pose_00_list = []
    pose_01_list = []
    pose_02_list = []

    #pose_10_list = []
    #pose_11_list = []
    #pose_12_list = []

    # Now, let the frames flow.
    while True:

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Step 1: Get a face from current frame.
        facebox = mark_detector.extract_cnn_facebox(frame)

        # Any face found?
        if facebox is not None:

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            # Run the detection.
            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.draw_annotation_box(
                frame, pose[0], pose[1], color=(0, 255, 0))

            # Do you want to see the head axes?
            pose_estimator.draw_axes(frame, pose[0], pose[1])
            print(pose[0][0][0])

            pose_00_list.append(pose[0][0][0]) #좌우
            pose_01_list.append(pose[0][1][0]) #위아래
            pose_02_list.append(pose[0][2][0]) #갸우뚱

            # 현재 해당 데이터 안쓰고 있음 첫 인덱스의 의미를 모르겠음

            #pose_10_list.append(pose[1][0])
            #pose_11_list.append(pose[1][1])
            #pose_12_list.append(pose[1][2])

            # Do you want to see the marks?
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # mark_detector.draw_box(frame, [facebox])

        # Show preview.
        cv2.imshow("Preview", frame)


        if cv2.waitKey(1) == 27:
            df0 = pd.DataFrame(pose_00_list)
            df1 = pd.DataFrame(pose_01_list)
            df2 = pd.DataFrame(pose_02_list)

            result3 = pd.concat([df0,df1,df2],axis=1)
            filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            result3.to_csv("02_pose_result\\"+filename+".csv") # 파일은 02_pose_result 에 csv 파일로 저장됩니다.

            break
