import cv2
import mediapipe as mp
import math
FRAME_DELAY = 10
mp_face_detection = mp.solutions.face_detection  # 얼굴 검출을 위한 face_detection 모듈을 사용
mp_drawing = mp.solutions.drawing_utils  # 얼굴의 특징을 그리기 위한 drawing_utils 모듈을 사용
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_hands = mp.solutions.hands
mp_fingers = mp_hands.HandLandmark
cap = cv2.VideoCapture(0)

def mosaic(image, x, y):
    x = int(x)
    y = int(y)
    x = max(min(x, 1819), 1)
    y = max(min(y, 979), 1)
    target_image = image[y: y + 100, x: x + 100]

    target_width, target_height, _ = target_image.shape

    target_image = cv2.resize(target_image, (0, 0), fx=0.02, fy=0.02)
    target_image = cv2.resize(target_image, (target_width, target_height), fx=0.02, fy=0.02, interpolation=cv2.INTER_AREA)

    image[y: y + 100, x: x + 100] = target_image
    return image


def isFuck(hand_landmarks):
    # return True
    # print(hand_landmarks[6].y, hand_landmarks[7].y)
    # print(hand_landmarks[14].y, hand_landmarks[15].y)
    # print(hand_landmarks[18].y, hand_landmarks[19].y)
    # print(hand_landmarks[10].y, hand_landmarks[11].y)

    if(hand_landmarks[6].y < hand_landmarks[7].y and
    hand_landmarks[14].y < hand_landmarks[15].y and
    hand_landmarks[18].y < hand_landmarks[19].y and
    hand_landmarks[10].y > hand_landmarks[11].y) :
        return True
    return False

def customConvertColor(image):
    for line in image:
        for dot in line:
            # print(dot)
            dot[0], dot[2] = dot[2], dot[0]
    return image


def overlay(image, x, y, w, h, overlay_image):  # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지
    alpha = overlay_image[:, :, 3]  # BGRA
    # print(alpha)
    # alpha = cv2.cvtColor(alpha, cv2.COLOR_BGRA2BGR)
    mask_image = alpha / 255  # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전)
    # print(mask_image)


    for c in range(0, 3):  # channel BGR
        image[y - h:y + h, x - w:x + w, c] = (overlay_image[:, :, c] * mask_image) + (
                    image[y - h:y + h, x - w:x + w, c] * (1 - mask_image))

image_right_eye = cv2.imread('left-removebg-preview.png', cv2.IMREAD_UNCHANGED) # 100 x 100
image_right_eye = customConvertColor(image_right_eye)
image_left_eye = cv2.imread('right-removebg-preview.png', cv2.IMREAD_UNCHANGED) # 100 x 100
image_left_eye = customConvertColor(image_left_eye)

hands = mp_hands.Hands(
        max_num_hands=10,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image,1)
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resultss = face_detection.process(image)
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if cv2.waitKey(1) == ord('q'):
            break
        width, height, _ = image.shape #1080 1920
        results = hands.process(image)

        if results.multi_hand_landmarks:
            # print(123123);
            for hand_landmarks in results.multi_hand_landmarks:
                if(isFuck(hand_landmarks.landmark)) :
                    for i in hand_landmarks.landmark:
                        image = mosaic(image, i.x*1920, i.y*1080)
                        
                    cv2.putText(
                        image,
                        text="Don't use 'FUCK YOU'",
                        org=(10,300),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=5.5,
                        color=(0,0,0),
                        thickness=30
                        )
                    if resultss.detections:
                        #얼굴 특징
                        for detection in resultss.detections:
                            # 특정 위치 가져오기
                            keypoints = detection.location_data.relative_keypoints
                            right_eye = keypoints[0]  # 오른쪽 눈
                            left_eye = keypoints[1]  # 왼쪽 눈
                            nose_tip = keypoints[2]  # 코 끝부분
                            
                            # 예외처리
                            if right_eye.y < 0.2: # 화면 모서리에 걸치면
                                continue 
                            if left_eye.y < 0.2: # 화면 모서리에 걸치면
                                continue
                            if right_eye.x > 0.8: # 화면 모서리에 걸치면
                                continue
                            if left_eye.x < 0.2: # 화면 모서리에 걸치면
                                continue
                            if nose_tip.y > 0.8: # 화면 모서리에 걸치면
                                continue

                            h, w, _ = image.shape  # height, width, channel : 이미지로부터 세로, 가로 크기 가져옴
                            right_eye = (int(right_eye.x * w) - 20, int(right_eye.y * h) - 100)  # 이미지 내에서 실제 좌표 (x, y)
                            left_eye = (int(left_eye.x * w) + 20, int(left_eye.y * h) - 100)
                            nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))


                            overlay(image, *right_eye, 50, 50, image_right_eye)
                            overlay(image, *left_eye, 50, 50, image_left_eye)
                            # overlay(image, *nose_tip, 150, 50, image_nose)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('MediaPipe Hands', image)
        # cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx=0.5, fy=0.5))
        cv2.waitKey(FRAME_DELAY)
        
cap.release()