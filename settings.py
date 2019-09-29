class Settings:
    input_dir = './input'  # 입력 디렉토리 위치. 서브디렉토리 구조로 되어있어야 합니다.
    output_dir = './output'  # 출력 위치
    face_threshold = 0.5  # 얼굴 인식 임계값. (0.0~1.0)
    license_plate_threshold = 0.8  # 번호판 인식 임계값. (0.0~1.0)
    bbox_red = 150  # 바운딩박스 빨간색 농도. (0~255)
    bbox_green = 150  # 바운딩박스 초록색 농도. (0~255)
    bbox_blue = 150  # 바운딩박스 파란색 농도. (0~255)
    bbox_thickness = 1  # 바운딩박스 외곽선 두께. (픽셀값)
    show_score = True  # True 이면 빨간색으로 클래스와 확신도를 표시합니다. False 이면 글자를 표시하지 않음.
    save_img = True  # True 이면 결과 이미지를 저장합니다. False 이면 이미지를 저장하지 않음.
    face_bbox_width_ratio = 0.6  # 얼굴 박스 너비 축소 비율입니다. (0.6이면 원래 크기의 60%로 줄입니다.)
    face_bbox_height_ratio = 0.6  # 얼굴 박스 너비 축소 비율입니다. (0.6이면 원래 크기의 60%로 줄입니다.)
    license_plate_bbox_width_ratio = 1.0  # 번호판 박스 너비 축소 비율.
    license_plate_bbox_height_ratio = 1.0  # 번호판 박스 너비 축소 비율.
    max_size_ratio = 0.33  # 크기가 너무 큰 물체는 필터링합니다. 0.33이면 전체 크기의 33%를 넘으면 무시합니다.
    license_plate_model_size = 704  # 모델이 학습된 크기입니다. 이 값은 바꾸지 마세요.
    xml_face_name = 'face'  # XML 파일에서 얼굴 태그의 라벨 이름을 지정합니다.
    xml_license_plate_name = 'license_plate'  # XML 파일에서 번호판 태그의 라벨 이름을 지정합니다.
