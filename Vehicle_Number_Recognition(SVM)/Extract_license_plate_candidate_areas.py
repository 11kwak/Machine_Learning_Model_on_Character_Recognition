import cv2
import numpy as np


# 전처리 함수 구현 : 명암도 영상 변환, 블러링 및 소벨 에지 검출 > 이진화 및 모폴로지 열림 연산 수행
def preprocessing(car_no):
    image = cv2.imread(
        "/Users/m1naworld/Desktop/ch07이미지파일_images/test_car/%02d.jpg" %
        car_no, cv2.IMREAD_COLOR)
    if image is None:
        print("이미지를 찾을 수 없습니다.")
        return None, None

    # 명암도 영상 변환, 블러링, 수직 에지 검출
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일
    gray = cv2.blur(gray, (5, 5))  # 블러링
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)  # 수직 에지 검출

    # 이진화 및 모폴로지 열림 연산 수행
    th_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]  # 이진화
    kernel = np.ones((5, 17), np.uint8)  # 가로로 긴 닫힘 연산 마스크
    morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel,
                             iterations=3)  # 닫힘 연산 3번 수행

    # 전처리 확인
    # cv2.imshow(
    #     "/Users/m1naworld/Desktop/ch07이미지파일_images/test_car/%02d.jpg" %
    #     car_no, morph)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image, morph


# 번호판 후보 영역 판정 함수
# 번호판 종횡비 검사
def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0:
        return False

    aspect = h / w if h > w else w / h
    chk1 = 1000 < (h * w) < 35000  # 번호판 넓이 조건 변경
    chk2 = 1.2 < aspect < 8.0  # 번호판 종횡비 조건 변경
    # chk1 = 3000 < (h * w) < 12000
    # chk2 = 2.0 < aspect < 6.5
    return chk1 and chk2  # bool 값


# 번호판 후보 검증
def find_candidates(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)  # 이진화 이미지에서 윤곽선 검색
    # cv2버전에 따라 findContours가 반환하는 값이 다른 것 같음
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rects = [cv2.minAreaRect(c) for c in contours]  # 회전 사각형 반환
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if verify_aspect_size(size)]

    return candidates


# 번호판 후보 영상 생성(후보영역 개선 -> 후보영역 재검증 -> 후보영상 회전 보정 -> 후보영상 생성)


# 후보영상 개선 함수 - 컬러 활용
def color_candidate_img(image, candi_center):
    h, w = image.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)  # 채움 행렬
    dif1, dif2 = (25, 25, 25), (25, 25, 25)  # 채움 색상 범위
    flags = 0xff00 + 4 + cv2.FLOODFILL_FIXED_RANGE  # 채움 방향 및 방법
    flags += cv2.FLOODFILL_MASK_ONLY  # 결과 영상만 채움

    ## 후보 영역을 유사 컬러로 채우기
    pts = np.random.randint(-15, 15, (20, 2))  # 임의 좌표 20개 생성
    pts = pts + candi_center  # 중심좌표로 평행이동
    for x, y in pts:  # 임의 좌표 순회
        if 0 <= x < w and 0 <= y < h:  # 후보 영역 내부 이면
            _, _, fill, _ = cv2.floodFill(image, fill, (x, y), 255, dif1, dif2,
                                          flags)
            return cv2.threshold(fill, 120, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow(
    #     "/Users/m1naworld/Desktop/ch07이미지파일_images/test_car/%02d.jpg", a)
    # cv2.waitKey(0)


# 후보영상 보정 함수5
def rotate_plate(image, rect):
    center, (w, h), angle = rect  # 중심 좌표, 크기, 회전각도
    if w < h:
        w, h = h, w
        angle = -1

    size = image.shape[1::-1]
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)  # 회전 행렬 계산
    rot_img = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)  # 회전 변환

    crop_img = cv2.getRectSubPix(rot_img, (w, h), center)  # 후보 영역 가져오기
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  # 명암도 영상
    return cv2.resize(crop_img, (144, 28))  # 크기변경 후 반환


car_no = int(input("자동차 영상 번호(0~17): "))
image, morph = preprocessing(car_no)  # 이미지 전처리
candidates = find_candidates(morph)  # 후보영상 생성

fills = [color_candidate_img(image, center)
         for center, _, _ in candidates]  # 후보 영역 재생성
new_candis = [find_candidates(fill) for fill in fills]  # 재생성 영역 검사
new_candis = [cand[0] for cand in new_candis if cand]  # 재후보 있으면 저장
candidate_imgs = [rotate_plate(image, cand)
                  for cand in new_candis]  # 후보 영상 각도 보정

for i, img in enumerate(candidate_imgs):
    cv2.polylines(image, [np.int32(cv2.boxPoints(new_candis[i]))], True,
                  (0, 255, 255), 2)
    cv2.imshow("candidate_img - " + str(i), img)

svm = cv2.ml.SVM_load("Open CV/SVMtrain.xml")
rows = np.reshape(candidate_imgs, (len(candidate_imgs), -1))
_, results = svm.predict(rows.astype("float32"))

correct = np.where(results == 1)[0]
# print('분류 결과:\n', results)
# print('번호판 영상 인덱스:', correct)

for i, idx in enumerate(correct):
    cv2.imshow("plate image_" + str(i), candidate_imgs[idx])
    cv2.resizeWindow("plate image_" + str(i), (250, 28))

for i, candi in enumerate(new_candis):
    color = (0, 255, 0) if i in correct else (0, 0, 255)
    cv2.polylines(image, [np.int32(cv2.boxPoints(candi))], True, color, 2)

print("번호판 검출완료") if len(correct) > 0 else print("번호판 미검출")
cv2.imshow("image", image)
cv2.waitKey(0)

## 계속 똑같이 출력 x 할때마다 출력이 다름 왜지????

#### 이미지에 따른 코드변환
# 4번사진 1/5확률로 판별
# 5번사진 def rotate_plate() angle값 -2로 변경시 번호판 판별 확률 ↑
# 7번 angle값 -5로 변경시 확률 ↑
# 9번 닫힘연산 2회 혹은 4회시 확률 ↑
# 10번 13번 잘안됨
# 14번 닫힘연산 2회시 확률 ↑
# 16번 닫힘연산 4회시 확률 ↑