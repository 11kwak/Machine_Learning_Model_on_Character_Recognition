import numpy as np, cv2

def calc_histo_hue(coin):
    hsv = cv2.cvtColor(coin, cv2.COLOR_BGR2HSV)             # 컬러 공간 변환
    hsize, ranges = [32], [0,180]                           # 32개 막대, 화소값 0~180 범위
    hist = cv2.calcHist([hsv], [0], None, hsize, ranges)    # 0(Hue)채널 히스토그램 계산
    return hist.flatten()                                   # 1차원 전개 후 반환

def grouping(hists):
    ws = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3,
          4, 5, 6, 8, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0]   # 가중치 32개 원소 지정

    sim = np.multiply(hists, ws)                            # 히스토그램과 가중치 곱
    similaritys = np.sum(sim, axis=1) / np.sum(hists, axis=1)   # 가중치 곱의 합/히스토그램 합
    groups = [1 if s > 1.2 else 0 for s in similaritys]

    ## 결과 보기
    # x = np.arange(len(ws))                                # 가중치 그래프 보기
    # plt.plot(x, ws, 'r'), plt.show(), plt.tight_layout()
    # for i, s in enumerate(similaritys):
    #       print('%d %5.0f  %d' % (i, s, groups[i]))
    return groups

def classify_coins(circles, groups):
    ncoins = [0] * 4
    coin_class = []
    g = np.full((2,70), -1, np.int)                         # 2행으로 두 개 그룹 설정
    g[0, 26:47], g[0, 47:50], g[0, 50:] = 0, 2, 3           # 10원 그룹- 10원 가능성 확대
    g[1, 36:44], g[1, 44:50], g[1, 50:] = 1, 2, 3           # 50원 그룹- 50원 100원 가능성 확대

    for group, (_, radius) in zip(groups, circles):         # 동전 객체 순회
        coin = g[group, radius]                             # 동전 종류 확정
        coin_class.append(coin)
        ncoins[coin] += 1                                   # 동전별 개수 산정
    return np.array(ncoins), coin_class                     # 넘파이 행렬로 반환