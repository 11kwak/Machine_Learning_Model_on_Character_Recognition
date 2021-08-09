from coin_preprocess import *
from coin_utils import *                                # 모든 함수 임포트
from Common.histogram import draw_hist_hue              # 색상 히스토그램 생성 함수

coin_no = 15                                            # 동전 번호
image, th_img = preprocessing(coin_no)                  # 전처리 수행
circles = find_coins(th_img)                            # 동전 객체 검출
coin_imgs = make_coin_img(image, circles)               # 동전 영상 분리
coin_hists = [calc_histo_hue(coin) for coin in coin_imgs]   # 색상 히스토그램

for i, img in enumerate(coin_imgs):
    h, w = 200, 256                                     # 히스토그램 영상 크기
    hist_img = draw_hist_hue(coin_hists[i], (h,w,3))    # 색상 히스토그램 표시

    merge = np.zeros((h,w+h,3), np.uint8)
    merge[:, :w] = hist_img                             # 결과 행렬 왼쪽 - 히스토그램 영상
    merge[:, w:] = cv2.resize(img, (h,h))               # 결과 행렬 오른쪽 - 동전 영상
    cv2.imshow('hist&coin - ' + str(i), merge)

cv2.waitKey(0)