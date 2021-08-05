def kNN_train(train_frame, K, nclass, nsample):
    size = (40, 40)
    train_img = cv2.imread(train_fname, cv2.IMREAD_GRAYSCLAE)
    h, w = train_img.shape[:2]
    dx = w % size[0] // 2
    dy = h % size[1] // 2
    train_img = train_img[dy:h-dy-1, dx:w-dx-1]
    cv2.threshold(train_img, 32, 255, cv2.THRESH_BINARY, train_img)
    cells = [np.hsplit(row, nsample) for row in np.vsplit(train_img, nclass)]
    nums = [find_numnber(c) for c in np.reshape(cells, (-1,40,40))]
    trainData = np.array([place_middle(n,size) for n in nums])
    labels = np.array([i for i in range(nclass) for j in range(nsample)], np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, labels)
    return knn

#데이터 전처리(번호판 영상의 크기, 180x35로 변경 후 이진화
#모서리 부분(좌우로 6픽셀, 상하 3픽셀)제거
def proprocessing_plate(plate_img):
    plate_img = cv2.resize(plate_img,(180,35))
    flag = cv2.THRESH_BINARY | cv2.THRESH_OTSU
    cv2.threshold(plate_img, 32, 255, flag, plate_img)
    
    h,w = plate_img.shpae[:2]
    dx, dy =(6,3)
    ret_img = plate_img[dy:h-dy, dx:w-dx]
    return ret_img

#숫자와 문자 객체 찾기 함수 find_object()로 구현
#검출 객체 넓이로 잡음 제거, 150보다 작은것은 잡음 
#검출 객체 종횡비로 잡음제거, 가로로 긴 객체는 숫자,문자아님 
#번호판은 숫자와 문자 공존
#번호판 영상의 크기가 일정하기 때문에 문자영역 위치 확인 가능, 45~80픽셀 범위객체는 문자로 인지
#문자객체 여러 부분으로 분리되어 검출 가능, 문자객체 영역들이 논리합으로 누적하여 하나의 영역으로 만듦, 문자객체 잡은은 넓이 60 미만 

def find_object(img):
    results = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROW_SIMPLE)
    contours = result[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rois = [cv2.boundingRect(contour) for contour in contours]
    rois = [(x,y,w,h,w*h) for x,y,w,h in rois if w/h<2.5]
    text_rois = [(x,y,x+w,y+h) for x,y,w,h,a in rois if 45<x<80 and a>60 ]
    num_rois = [(x,y,w,h) for x,y,w,h,a in rois if not(45<x<80) and a>150]

    if text_rois:                                #분리된 문자 영역 누적
        pts = np.sort(atext_rois,axis=0)        #세로방향 정렬
        x0,y0 = pts[0, 0:2]                     #시작좌표 중 최소인 x,y 좌표
        x1,y1 = pts[-1,2:]                      #종료좌표 중 최대인 x,y 좌표
        w,h = x1-x0, y1-y0                      #너비, 높이 계산
        num_rois.append((x0,y0,w,h))            #문자 영역 구성 및 저장
    return num_rois

#검출된 객체 영상의 숫자 및 문자를 인식하기 
#숫자와 문자 사각형들 벡터(object_rects)에 저장 
#cv2.findContours() 함수로 객체 외각선 검출시
#검출 사각형들 위치 정렬되어 있지 않음 -> 각 사각형이 몇 번째 숫자/문자 인지 확인불가
#검출 객체를 x 좌표 순으로 정렬 필요, np.argsort() 함수 사용하여 정렬 인덱스 사용

#검출 객체 영상 분류하여 숫자 및 문자 인식
#classify_numbers() 함수로 구현 
#10장 3절 숫자 인식 예제에서 사용한 함수 사용
#find_number() 함수-숫자 객체 검출, place_middle() 함수-숫자 중앙배치 및 1행 데이터 생성

def classify_numbers(cells, nknn, tknn, k1, k2, object_rois):
    if len(cells) != 7:
        print("검출된 숫자(문자)가 7개가 아닙니다.")
        return 

    texts = "가나다라마거너더러머버서어저고노도로모보"\
            "소오조구누두루무부수우주아바사자바하허호"
    numbers = [find_number(cell) for cell in cells]
    datas = [place_middle(num, (40,40)) for num in numbers]
    datas = np.reshape(datas, (len(datas), -1))

    idx = np.argsort(object_rois, axis=0).i[0]
    text = datas[idx[2]].reshape(1,-1)
    _, resp1, _, _ = nknn.findNearest(datas, K1)
    _, [[resp2]], _, _ = tknn.findNearest(text, K2)

    resp1 = resp1.flatten().astype("int")
    results = resp1[idx].astype("str")
    results[2] = texts[int(resp2)]

    print("정렬 인덱스 :", idx)
    print("숫자 객체 결과 :", resp1)
    print("문자 객체 결과 :", int(resp2))
    print("분류 결과 :","".join(results))
    
