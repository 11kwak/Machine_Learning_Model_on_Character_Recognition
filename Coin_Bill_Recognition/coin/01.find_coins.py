from coin_preprocess import *                           # 전처리 함수 임포트

image, th_img = preprocessing(70)                       # 전처리 수행
if image is None: raise Exception('영상파일 읽기 에러')

circles = find_coins(th_img)                            # 동전 객체 검출
for center, radius in circles:
    cv2.circle(image, center, radius, (0,255,0), 2)     # 영상에 검출 원 표시

cv2.imshow('preprocessed image', th_img)
cv2.imshow('coin image', image)
cv2.waitKey(0)