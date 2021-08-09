import sys
import os
import numpy as np
import cv2


# 변수설정 ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
###################################################################################################


def main():
    # 학습할 사진 불러오기
    imgTrainingNumbers = cv2.imread("training_chars.png")

    if imgTrainingNumbers is None:
        print("error: image not read from file \n\n")
        # 경고 메세지를 볼 수 있도록 정지
        os.system("pause")
        return

    # grayscale image : BGR(blue + green + red)을 GRAY로 색상을 바꾸는 작업
    # Blur 처리: 이미지의 노이즈를 제거하는 작업
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(
        imgGray, (5, 5), 0)

    # Binary Image : grayscale로 변환한 이미지를 흑백(이진화) 이미지로 바꾸는 작업
    # cv2.adaptiveThreshold(입력 이미지, 최댓값, 적응형 이진화 플래그, 임곗값 형식, 블록 크기, 감산값)
    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                      255,    # 픽셀을 threshold(반올림)해서 다 하얀색으로
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      11,
                                      2)

    # threshold image 가 어떻게 나왔는지 띄어주는것
    cv2.imshow("imgThresh", imgThresh)

    # 이미지를 복사해서 윤곽선을 수정해주고, 윤곽변수설정
    imgThreshCopy = imgThresh.copy()
    npaContours, npaHierarchy = cv2.findContours(
        imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 글자로 인식되어 캡쳐될 npaFlattenedImages 를 넘파이 형식으로 변수설정
    npaFlattenedImages = np.empty(
        (0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    # 문자를 분류하는 방법 시작. 마지막 파일에 쓸 것, 문자분류목록 만듬
    intClassifications = []

    # 문자의 유니코드 값을 돌려받아서 intValidChars에 리스트화 시켜놓기
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord(
                         'F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord(
                         'P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:                   # 각 윤곽에 대해
        # 윤곽이 충분히 크다면
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(
                npaContour)         # bounding rect함수는 인자로 받은 contour에 외접하고 똑바로 세워진 직사각형의 최상단 꼭지점 좌표 가로 세로 폭을 리턴, 좌표를 이용해 원본 이미지에 빨간색 표시

            # 유저가 넣을 글자를 직사각형으로 보여줌
            cv2.rectangle(imgTrainingNumbers,           # draw rectangle on original training image
                          (intX, intY),                 # upper left corner
                          (intX+intW, intY+intH),        # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            # 문자 자르기
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]
            # 자른 문자의 이미지의 사이즈 조절
            imgROIResized = cv2.resize(
                imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            # 문자 자른거 보여주기
            cv2.imshow("imgROI", imgROI)
            # 문자 자른거 사이즈 조절한거 보여주기
            cv2.imshow("imgROIResized", imgROIResized)
            # 문자 학습 이미지 보여주기, 빨간 직사각형 보일 이미지임.
            cv2.imshow("training_numbers.png", imgTrainingNumbers)

            intChar = cv2.waitKey(0)                # get key press

            if intChar == 27:                   # esc key가 눌리면
                sys.exit()                      # exit program
            # 문자 리스트에 우리가 입력한게 있으면 . . .
            elif intChar in intValidChars:

                # 문자분류목록에 입력한 유니코드 추가
                intClassifications.append(intChar)

                # 캡쳐된 이미지를 numpy 배열로 쓸수 있도록  변환
                npaFlattenedImage = imgROIResized.reshape(
                    (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                # 넘파이 배열로 변환한 캡쳐된 이미지를 npaFlattenedImages에 추가
                npaFlattenedImages = np.append(
                    npaFlattenedImages, npaFlattenedImage, 0)

    # 유니코드를 실수형 넘파이 배열로변환
    fltClassifications = np.array(intClassifications, np.float32)

    # 넘파이배열로 변환한 유니코드를 사이즈 조정
    npaClassifications = fltClassifications.reshape(
        (fltClassifications.size, 1))

    print("\n\ntraining complete !!\n")

    # 파일화 시키기
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)

    cv2.destroyAllWindows()             # 메모리 제거

    return


###################################################################################################
if __name__ == "__main__":  # 다른데서 실행못하게, 정보은닉의 확장판
    main()
