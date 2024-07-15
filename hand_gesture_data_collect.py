import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# Mediapipe 손 인식 모델 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 제스처 리스트
gestures = ['V', 'thumbs_up', 'hand_open', 'fist', 'heart', 'gun', 'one']

# 데이터프레임 초기화 (중요 랜드마크만 사용)
landmark_indices = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17]
columns = [f'x{i}' for i in landmark_indices] + [f'y{i}' for i in landmark_indices] + ['gesture']
df = pd.DataFrame(columns=columns)

# 각 제스처에 대한 데이터를 수집하는 함수
def collect_gesture_data(gesture_name):
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # for mac, WSL2
    frame_count = 0
    save_count = 0
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
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
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 랜드마크 추출 (중요 랜드마크만)
                    landmarks = []
                    for i in landmark_indices:
                        lm = hand_landmarks.landmark[i]
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                    landmarks = np.array(landmarks).flatten()

                    # 데이터프레임에 추가
                    row = list(landmarks) + [gesture_name]

                    # 현재 제스처와 저장된 이미지 개수 화면에 표시
                    cv2.putText(image, f'gesture: {gesture_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'saved: {save_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    global df
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        df = pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)
                        save_count += 1

            cv2.imshow(f'{gesture_name} gesture', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

# 제스처 데이터 수집
for gesture in gestures:
    print(f'{gesture} 수집 시작')
    collect_gesture_data(gesture)
    print(f'{gesture} 수집 완료.')

# 데이터프레임 저장
df.to_csv(f'./gesture_da1ta.csv', index=False)
print('데이터 수집 완료! : gesture_data.csv.')
