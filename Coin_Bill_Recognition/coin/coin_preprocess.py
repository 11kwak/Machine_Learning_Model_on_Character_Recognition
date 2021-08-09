import numpy as np, cv2

def preprocessing(coin_no):                                 # 전처리 함수
    fname = './coin/{0:02d}.png'.format(coin_no)
    image = cv2.imread(fname, cv2.IMREAD_COLOR)             # 영상 읽기
    if image is None: return None, None                     # 예외처리는 메인에서

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # 명암도 영상 변환
    gray = cv2.GaussianBlur(gray, (7,7), 2, 2)              # 블러링
    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU              # 오츠(otus) 이진화 지정
    _, th_img = cv2.threshold(gray, 130, 255, flag)         # 이진화

    mask = np.ones((3,3), np.uint8)
    th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, mask) # 열림 연산
    return image, th_img

def find_coins(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    circles = [cv2.minEnclosingCircle(c) for c in contours] # 외각 감싸는 원 검출
    circles = [(tuple(map(int, center)), int(radius))
               for center, radius in circles if radius>25]
    return circles

def make_coin_img(src, circles):
    coins = []
    for center, radius in circles:
        r = radius * 3                                      # 검출 동전 반지름 3배
        cen = (r // 2, r // 2)                              # 마스크 중심
        mask = np.zeros((r,r,3), np.uint8)                  # 마스크 행렬
        cv2.circle(mask, cen, radius, (255,255,255), cv2.FILLED)
        # cv2.imshow('mask_' + str(center), mask)           # 마스크 영상 보기

        coin = cv2.getRectSubPix(src, (r,r), center)        # 동전 영상 가져오기
        coin = cv2.bitwise_and(coin, mask)                  # 마스킹 처리
        coins.append(coin)                                  # 동전 영상 저장장
    return coins