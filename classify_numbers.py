from Vehicle_Number_Recognition.defs import kNN_train, proprocessing_plate
import defs as d


from plate_preprocess import *
from plate_candidate import *
from plate_classify import *

car_no = int(input("자동차 영상 번호(0~15): "))
image, morph = preprocessing(0)
candidates = find_candidates(morph)

fills = [refine_candidate_img(image,size) for size, _, _ in candidates ]
new_candi = [find_candidates(fill) for fill in fills]
new_candi = [cand[0] for cand in new_candi if cand]
candidate_imgs = [rotate_plate(image, cand) for cand in new_candi]

svm = cv2.ml.SVM_load("SVMTrain.xml")
rows = np.reshape(candidate_imgs, (len(candidate_imgs), -1))
_, results = svm.predict(rows.astype("float32"))
result = np.where(results.flatten() == 1)[0]

plate_no = result[0] if len(result) > 0 else -1

K1, K2 = 10, 10
nknn = kNN_train("images/train_numbers.png",K1,10,20)
tknn = kNN_train("images/train_texts.png",K2,40,20)

if plate_no >= 0:
    plate_img = proprocessing_plate(candidate_imgs[plate_no])
    cells_roi = find_objects(cv2.bitwise_not(plate_img))
    cells = [plate_img[y:y+h, x:x+w] for x,y,w,h in cells_roi]

    classify_numbers(cells. nknn, tknn, K1, K2, cells_roi)

    pts = np.int32(cv2.cvtColor(candidatesp[plate_no]))
    cv2.polylines(image, [pts], True, (0,255,0),2)

    color_plate = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)
    for x,y,w,h in cells_roi:
        cv2.rectangle(color_plate, (x,y), (x+w, y+h), (0,0,255),1)

    h,w = color_plate.shape[:2]
    image[0:h, 0:w] = color_plate

else:
    print("번호판 미검출")

cv2.imshow("image",image)
cv2.waitkey(0)





