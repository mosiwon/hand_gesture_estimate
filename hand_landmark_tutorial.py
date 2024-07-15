import cv2
import mediapipe as mp

# Mediapipe 손 인식 모델 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 웹캠 초기화
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # for mac, WSL2

with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 성능을 향상시키기 위해 이미지 쓰기 금지 설정
        image.flags.writeable = False

        # 이미지 처리
        results = hands.process(image)

        # 이미지 쓰기 허용 설정
        image.flags.writeable = True

        # RGB 이미지를 다시 BGR로 변환
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크 그리기
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 이미지 출력
        cv2.imshow('Hand Landmarks', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 웹캠 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()