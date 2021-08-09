import cv2
import numpy as np

def color_candidate_image(image, center):
    h, w = image.shape[:2]
    fill = np.zeros((h+2, w+2), np.uint8)                                 # 채움 행렬
    dif1, dif2 = (25,25,25), (25,25,25)                                     # 채움 색상 범위
    flags = 0xff00 + 4 + cv2.FLOODFILL_FIXED_RANGE                          # 채움 방향 및 방법
    flags += cv2.FLOODFILL_MASK_ONLY                                        # 결과 영상만 채움

    ## 후보 영역을 유사 컬러로 채우기
    pts = np.random.randint(-15, 15, (20,2))                                # 임의 좌표 20개 생성
    pts = pts + center                                                      # 중심 좌표로 평행이동
    for x, y in pts:                                                        # 임의 좌표 순회
        if 0 <= x < w and 0 <= y < h:                                       # 후보 영역 내부이면
            _, _, fill, _ = cv2.floodFill(image, fill, (x, y), 255, dif1, dif2, flags)  # 채움 누적

        return cv2.threshold(fill, 120, 255, cv2.THRESH_BINARY)[1]

def move_num(image, rect):
    center, (w, h), angle = rect                                            # 중심 좌표, 크기, 회전 각도
    if w < h :                                                              # 세로가 긴 영역이면
        w, h = h, w                                                         # 가로와 세로 맞바꿈
        angle += 90                                                         # 회전 각도 조정

    size = image.shape[1::-1]                                               # 행태와 크기는 역순
    mov_mat = np.float32([[1,0,8], [0,1,0]])                                 # x,y 이미지 이동 계산
    mov_img = cv2.warpAffine(image, mov_mat, size)                          # 이동 변환

    crop_img = cv2.getRectSubPix(mov_img, (w, h), center)                   # 후보 영역 가져오기
    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)                   # 명암도 영상
    return crop_img

