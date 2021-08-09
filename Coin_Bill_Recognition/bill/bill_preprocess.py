import numpy as np, cv2

def show(img):
    cv2.imshow('test', img)
    cv2.waitKey(0)

def preprocessing(bill_no):
    image = cv2.imread('./bill/%02d.jpg' % bill_no, cv2.IMREAD_COLOR)
    if image is None: return None, None
    image = cv2.resize(image, (640,480), interpolation=cv2.INTER_AREA)

    kernel = np.ones((7, 18), np.uint8)                                     # 모폴로지 연산 마스크(커널)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                          # 명암도 영상 변환
    gray = cv2.GaussianBlur(gray, (7,7), 0)                                 # 블러링
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)                              # 수직 에지 검출

    _, th_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)            # 이진화 수행
    morph = cv2.dilate(th_img, kernel, iterations=1)

    # cv2.imshow('th_img', th_img); cv2.imshow('morph', morph)              # 결과표시
    return image, morph

def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h / w if h > w else w / h                                      # 세로가 길면 역수 취함
    chk1 = 3000 < (h*w) < 12000                                             # 숫자 넓이 조건
    chk2 = 2.0 < aspect < 6.5                                               # 숫자 종횡비 조건
    return (chk1 and chk2)

def find_candidates(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rects = [cv2.minAreaRect(c) for c in contours]                          # 회전 사각형 반환
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)   # 정수형 반환
              for center, size, angle in rects if verify_aspect_size(size)]

    return candidates

