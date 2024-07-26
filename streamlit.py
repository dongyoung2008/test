
import streamlit as st
import cv
import numpy as np
from PIL import Image
import openvino.runtime as ov
import tempfile
import os
import time

# OpenVINO 모델 로딩
core = ov.Core()

# 모델 파일 경로
model_face = core.read_model(model="face-detection-adas-0001.xml")
compiled_model_face = core.compile_model(model=model_face, device_name="CPU")
input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

model_emo = core.read_model(model="emotions-recognition-retail-0003.xml")
compiled_model_emo = core.compile_model(model=model_emo, device_name="CPU")
input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

model_ag = core.read_model(model="age-gender-recognition-retail-0013.xml")
compiled_model_ag = core.compile_model(model=model_ag, device_name="CPU")
input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output(0)

def preprocess(image, input_layer):
    N, input_channels, input_height, input_width = input_layer.shape
    resized_image = cv.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    return input_image

def find_faceboxes(image, results, confidence_threshold):
    results = results.squeeze()
    scores = results[:, 2]
    boxes = results[:, -4:]
    face_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    image_h, image_w, _ = image.shape
    face_boxes = face_boxes * np.array([image_w, image_h, image_w, image_h])
    face_boxes = face_boxes.astype(np.int64)
    return face_boxes, scores

def draw_faceboxes(image, face_boxes):
    show_image = image.copy()
    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        cv.rectangle(img=show_image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 200, 0), thickness=2)
    return show_image

def draw_emotions_and_age_gender(face_boxes, image):
    show_image = image.copy()
    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        face = image[ymin:ymax, xmin:xmax]

        # 감정 인식
        input_image_emo = preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_image_emo])[output_layer_emo]
        results_emo = results_emo.squeeze()
        index = np.argmax(results_emo)
        emotion = EMOTION_NAMES[index]

        # 나이 및 성별 인식
        input_image_ag = preprocess(face, input_layer_ag)
        results_ag = compiled_model_ag([input_image_ag])
        age = np.squeeze(results_ag[1]) * 100
        gender = np.squeeze(results_ag[0])
        if gender[0] >= 0.65:
            gender = "female"
            box_color = (255, 153, 255)
        elif gender[1] >= 0.65:
            gender = "male"
            box_color = (255, 204, 154)
        else:
            gender = "unknown"
            box_color = (0, 0, 0)

        fontScale = image.shape[1] / 750
        text = f"{gender} {int(age)} {emotion}"
        cv.putText(show_image, text, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, fontScale, box_color, 2)
        cv.rectangle(img=show_image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=box_color, thickness=2)
    return show_image

def resize_frame(frame, scale=0.5):
    height, width = frame.shape[:2]
    new_dim = (int(width * scale), int(height * scale))
    resized_frame = cv.resize(frame, new_dim, interpolation=cv.INTER_AREA)
    return resized_frame

# Streamlit 앱 시작
st.sidebar.title('메뉴')
option = st.sidebar.radio('메뉴 선택', ['Photo', 'Video', 'Webcam', 'GitHub 프로필'])

if option == 'GitHub 프로필':
    st.title('GitHub 프로필')
    st.markdown("[GitHub 프로필로 이동하기](https://github.com/dongyoung2008/openvino.git)")

if option == 'Photo':
    st.title('Photo 메뉴')
    uploaded_file = st.file_uploader('사진을 업로드하세요', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # PIL 이미지를 OpenCV 이미지로 변환
        image = Image.open(uploaded_file).convert('RGB')  # Ensure the image is in RGB format
        image = np.array(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        
        input_image = preprocess(image, input_layer_face)
        results = compiled_model_face([input_image])[output_layer_face]
        
        confidence_threshold = 0.95
        face_boxes, scores = find_faceboxes(image, results, confidence_threshold)
        show_image = draw_faceboxes(image, face_boxes)
        show_image = draw_emotions_and_age_gender(face_boxes, show_image)
        
        # Convert BGR image to RGB for Streamlit display
        show_image = cv.cvtColor(show_image, cv.COLOR_BGR2RGB)
        st.image(show_image, caption='얼굴이 감지된 사진', channels='RGB')

if option == 'Video':
    st.title('Video 메뉴')
    uploaded_video = st.file_uploader('비디오를 업로드하세요', type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        # 비디오 파일을 임시 디렉토리에 저장
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(tfile.name, 'wb') as f:
            f.write(uploaded_video.read())
        
        # 비디오 처리
        stframe = st.empty()
        cap = cv.VideoCapture(tfile.name)
        
        fps = cap.get(cv.CAP_PROP_FPS)
        delay = 1 / fps  # 프레임 간 시간
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # 비디오를 처음부터 다시 시작
                continue
            
            # 프레임 해상도 조정
            frame = resize_frame(frame, scale=0.5)  # 해상도를 50%로 줄임
            
            # 얼굴 감지 및 감정, 나이, 성별 인식
            input_image = preprocess(frame, input_layer_face)
            results = compiled_model_face([input_image])[output_layer_face]
            
            confidence_threshold = 0.95
            face_boxes, scores = find_faceboxes(frame, results, confidence_threshold)
            frame = draw_faceboxes(frame, face_boxes)
            frame = draw_emotions_and_age_gender(face_boxes, frame)
            
            # 비디오 프레임 표시
            stframe.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), channels='RGB', use_column_width=True)
            
            # 프레임 간 시간 조절 (FPS 조절)
            time.sleep(delay)  # FPS에 맞춰 지연 시간 조절
            
        cap.release()
        os.remove(tfile.name)

if option == 'Webcam':
    st.title('Webcam 메뉴')

    if st.button('Start'):
        stframe = st.empty()
        cap = cv.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("웹캠에서 프레임을 읽을 수 없습니다.")
                break
            
            # 프레임 해상도 조정
            frame = resize_frame(frame, scale=0.5)  # 해상도를 50%로 줄임

            # 얼굴 감지 및 감정, 나이, 성별 인식
            input_image = preprocess(frame, input_layer_face)
            results = compiled_model_face([input_image])[output_layer_face]
            
            confidence_threshold = 0.95
            face_boxes, scores = find_faceboxes(frame, results, confidence_threshold)
            frame = draw_faceboxes(frame, face_boxes)
            frame = draw_emotions_and_age_gender(face_boxes, frame)
            
            # 웹캠 프레임 표시
            stframe.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), channels='RGB', use_column_width=True)

        cap.release()
