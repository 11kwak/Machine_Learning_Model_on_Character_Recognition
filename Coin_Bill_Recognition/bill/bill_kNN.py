import cv2
import numpy as np
import operator
import os
# 지폐이미지 가져오기 ######################################################################
from bill_preprocess import *                                           # 전처리 및 후보 영역 검출 함수
from bill_candidate import *                                            # 후보 영역 개선 및 후보 영상 생성 함수



# 변수설정 ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
###################################################################################################


class ContourWithData():

    # 클래스 변수초기화 ############################################################################
    npaContour = None           # 윤곽
    boundingRect = None         # 윤곽에 대한 경계 직선
    intRectX = 0                # 경계 직사각형 왼쪽 상단 모서리 x 위치
    intRectY = 0                # 경계 직사각형 오른쪽 상단 모서리 x 위치
    intRectWidth = 0            # 경계 직사각형 폭
    intRectHeight = 0           # 경계 직사각형 높이
    fltArea = 0.0               # 윤곽 면적

    # 윤곽경계 계산하기
    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    # 유효한 자료인지 면적 비교를 통해 확인
    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA:
            return False
        return True

###################################################################################################


def main():
    allContoursWithData = []                # 비어있는 리스트 선언
    validContoursWithData = []              # 유효한거 넣는 비어있는 리스트 선언

    try:
        # 넘파이로 만든 유니코드 텍스트 불러오기
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return

    try:
        # 넘파이로 만든 이미지들 불러오기
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return

    # 넘파이 배열을 1차원으로 바꾸기, 트레이닝하려면 필수
    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # 테스트 이미지가 무엇인지 설정
    ##########################################################################
    bill_no = int(input('지폐 영상 번호(0~2): '))
    image, morph = preprocessing(bill_no)  # 전처리- 소벨&모폴로지 연산
    candidates = find_candidates(morph)  # 숫자 후보 영역 검색
    candidate_imgs = [move_num(image, cand) for cand in candidates]  # 후보 영역 저장

    elect_img = candidate_imgs[0]  # 최종 숫자 선택
    cv2.GaussianBlur(elect_img, (5, 5), 0)  # 전처리- 이진화&모폴로지
    # cv2.threshold(elect_img, 110, 255, cv2.THRESH_BINARY, elect_img)
    # kernel = np.ones((1, 1), np.uint8)
    # morph = cv2.morphologyEx(elect_img, cv2.MORPH_DILATE, kernel, iterations=1)
    ###########################################################################
    imgTestingNumbers = elect_img

    if imgTestingNumbers is None:                           # 이미지 안불러지면

        print("error: image not read from file \n\n")
        # 유저가 알 수 있게 정지
        os.system("pause")
        return

    # grayscale image : BGR(blue + green + red)을 GRAY로 색상을 바꾸는 작업
    # Blur 처리: 이미지의 노이즈를 제거하는 작업
    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)                    # blur

    # Binary Image : grayscale로 변환한 이미지를 흑백(이진화) 이미지로 바꾸는 작업
    # cv2.adaptiveThreshold(입력 이미지, 최댓값, 적응형 이진화 플래그, 임곗값 형식, 블록 크기, 감산값)
    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                      255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      11,
                                      2)

    # 이미지를 복사해서 윤곽선을 수정해주고, 윤곽변수설정
    imgThreshCopy = imgThresh.copy()

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # 입력 이미지, 윤곽선을 찾는 과정에서 이미지를 수정하기떄문에 복사본을 사용
                                                 cv2.RETR_EXTERNAL,         # 가장 바깥쪽 윤곽만 검색
                                                 cv2.CHAIN_APPROX_SIMPLE)   # 수평, 수직 및 대각선 부분을 압축하고 끝점만 남김

    for npaContour in npaContours:                             # 각 윤곽마다
        # 데이터 객체를 사용하여 윤곽 인스턴스화
        contourWithData = ContourWithData()
        # 데이터를 사용하여 윤곽에 데이터 할당
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(
            contourWithData.npaContour)     # 경계선을 정확히 잡기
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight(
        )                    # 경계선에 맞는 정보 가져오기
        contourWithData.fltArea = cv2.contourArea(
            contourWithData.npaContour)           # 윤곽면적계산
        # 데이터 객체가 있는 윤곽을 데이터가 있는 모든 윤곽 리스트에 추가
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:                 # 모든육관을
        if contourWithData.checkIfContourIsValid():             # 유효한지확인하고
            # 맞다면 유효한 윤곽리스트에 넣기
            validContoursWithData.append(contourWithData)

    validContoursWithData.sort(key=operator.attrgetter(
        "intRectX"))         # 윤곽을 왼쪽에서 오른쪽으로 정렬

    # 문자를 반환하는 strFinalString 초기화
    strFinalString = ""

    for contourWithData in validContoursWithData:            # 모든 윤곽마다
        # 현재의 문자 주위에 녹색점을 그리다.
        cv2.rectangle(imgTestingNumbers,                                        # 원본 테스트 이미지에 사각형 그리기
                      # 위 왼쪽 코너
                      (contourWithData.intRectX, contourWithData.intRectY),
                      (contourWithData.intRectX + contourWithData.intRectWidth,
                       contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # 초록색
                      2)                        # 얇게

        imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

        # 인식 및 저장에 대해 보다 일관성이 있게 하려고 이미지 크기 조정
        imgROIResized = cv2.resize(
            imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        # 조정한 이미지를 1차원배열로 reshape
        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        # 1차원 배열을 실수형으로 만들기
        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(
            npaROIResized, k=1)     # knn이용해서 가장 인접한 글자찾기

        # 결과값에 지금 문자 넣기
        strCurrentChar = str(chr(int(npaResults[0][0])))

        # 문장형으로 만들 수 있도록 글자 합쳐지게 하기
        strFinalString = strFinalString + strCurrentChar

    print("\n" + strFinalString + '원 입니다.' + "\n")                  # 문장 보여지게하기

    # 지폐 & 영역 이미지 표시
    pts = np.int32(cv2.boxPoints(candidates[0]))
    cv2.polylines(image, [pts], True, (0, 255, 255), 2)  # 다중 좌표 잇기
    cv2.imshow('imgBillNumbers', image)
    # 찾은 글자 주위에 녹색 상자가 그려진 입력 이미지 표시
    cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    # 유저가 키 입력할때까지 무한대기
    cv2.waitKey(0)

    cv2.destroyAllWindows()             # 메모리 초기화

    return


###################################################################################################
if __name__ == "__main__":
    main()
