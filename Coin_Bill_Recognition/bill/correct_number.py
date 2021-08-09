import cv2
from bill_preprocess import *                                           # 전처리 및 후보 영역 검출 함수
from bill_candidate import *                                            # 후보 영역 개선 및 후보 영상 생성 함수

car_no = 0
image, morph = preprocessing(car_no)                                    # 전처리- 소벨&모폴로지 연산
candidates = find_candidates(morph)                                     # 숫자 후보 영역 검색

# fills = [color_candidate_image(image, size) for size, _, _ in candidates]   # 후보 영역 재생성
# new_candis = [find_candidates(fill) for fill in fills]                  # 재생성 영역 검사
# new_candis = [cand[0] for cand in new_candis if cand]                   # 재후보 있으면 저장
candidate_imgs = [move_num(image, cand) for cand in candidates]         # 후보 영역 저장

#for i, img in enumerate(candidate_imgs):                                # 후보 영상 표시
#    cv2.polylines(image, [np.int32(cv2.boxPoints(candidates[i]))], True, (0,255,255), 2)
#    cv2.imshow('candidate_img - ' + str(i), img)
elect_img = candidate_imgs[0]
cv2.GaussianBlur(elect_img, (5,5), 0)
cv2.threshold(elect_img, 110, 255, cv2.THRESH_BINARY, elect_img)
kernel = np.ones((1, 1), np.uint8)
morph = cv2.morphologyEx(elect_img, cv2.MORPH_DILATE, kernel, iterations=1)

cv2.imshow('image', morph)
cv2.waitKey(0)
