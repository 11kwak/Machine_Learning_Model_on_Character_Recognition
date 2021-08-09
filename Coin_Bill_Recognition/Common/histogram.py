import numpy as np, cv2


def make_palette(rows):                             # hue 채널 팔레트 행렬 생성 함수
    ## 리스트 생성 방식
    hue = [round(i*180/rows) for i in range(rows)]  # hue 값 리스트 계산
    hsv = [[[h,255,255]] for h in hue]              # (hue,255,255) 화소값 계산
    hsv = np.array(hsv, np.uint8)                   # 정수(uint8)형 행렬 변환
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)     # HSV 컬러 -> BGR 컬러

def draw_hist_hue(hist, shape=(200,256,3)):         # 색상 히스토그램 그리기 함수
    hsv_palette = make_palette(hist.shape[0])       # 색상 팔레트 생성
    hist_img = np.full(shape, 255, np.uint8)
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX) # 영상 높이값으로 정규화

    gap = hist_img.shape[1] / hist.shape[0]         # 한 계급 크기
    for i, h in enumerate(hist):
        x, w = int(round(i * gap)), int(round(gap))
        color = tuple(map(int, hsv_palette[i][0]))  # 정수형 튜플로 변환
        cv2.rectangle(hist_img, (x,0, w,int(h)), color, cv2.FILLED) # 팔레트 색으로 그림

    return cv2.flip(hist_img, 0)