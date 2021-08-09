from bill_preprocess import *                                           # 전처리 및 후보 영역 검사 함수

bill_no = int(input('지폐 영상 번호(0~2): '))
image, morph = preprocessing(bill_no)                                   # 전처리- 소벨&모폴로지 연산
if image is None: Exception('영상파일 읽기 에러')

candidates = find_candidates(morph)                                     # 숫자 후보 영역 검색
for candidate in candidates:                                            # 후보 영역 표시
    pts = np.int32(cv2.boxPoints(candidate))
    cv2.polylines(image, [pts], True, (0,255,255), 2)                   # 다중 좌표 잇기
    print(candidate)

if not candidates:                                                      # 리스트 원소가 없으면
    print('숫자 후보 영역 미검출')
cv2.imshow('image', image)
cv2.waitKey(0)