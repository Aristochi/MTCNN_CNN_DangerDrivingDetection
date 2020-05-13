
import os
import random
import time
import  numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.saving import load_model
from keras.preprocessing.image import img_to_array

from mobileNet import MobileNet
from mtcnn import MTCNN
from math import *
detector = MTCNN()





# cap = cv2.VideoCapture(0)
def get_MER(x1,x2,y1,y2,img):
    cropped = img[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]
    return cropped



modelpath='./model/best0428ep150.h5'
model_cnn=load_model(modelpath, compile=False)
classname=["closed_eye","closed_mouth","open_eye","open_mouth","smoke"]
def get_label(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    preds = model_cnn.predict(img)
    i = preds.argmax(axis=1)[0]
    label = classname[i]

    return label

# img_count=0
left_eye_count=0
right_eye_count=0
mouth_count=0
frag=False
blink=0
path='./20200407_173126.mp4'

cap = cv2.VideoCapture(path)

start = time.time()
while True:
    start = time.time()
    red,image=cap.read()
    Img=image.copy()

    # image = cv2.resize(image, (480, 360))
    result = detector.detect_faces(image)
    if(len(result))>0:



    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    # print(len(result[0]['box']))
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']
        #
        # print(face)
        left_eye=keypoints['left_eye']
        right_eye=keypoints['right_eye']
        nose=keypoints['nose']
        mouth_left=keypoints['mouth_left']
        mouth_right=keypoints['mouth_right']
        arc = atan(abs(right_eye[1] - left_eye[1]) / abs(right_eye[0] - left_eye[0]))
        W = abs(right_eye[0] - left_eye[0]) / (2 * cos(arc))
        H = W / 2
    ###########可去掉，只是避免裁剪出问题而已
        x1 = int(left_eye[0] - W / 2)
        if(x1<=0):
            x1=1
        x2 = int(left_eye[0] + W / 2)
        if(x2>=639):
            x2=638
        y1 = int(left_eye[1] - H / 2)-5
        if (y1<=0):
            y1=1
        y2 = int(left_eye[1] + H / 2)
        if(y2>=479):
            y2=478

        left=get_MER(x1, x2, y1, y2, image)
        label_left_eye = get_label(left)
        if label_left_eye=='closed_eye':
            left_state='closed'
        else:
            left_state='open'
        print(label_left_eye)
        cv2.putText(image, "{}".format(left_state), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0),
                     1, 8)
        cv2.putText(image, "left_eye_state:{}".format(left_state), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, 8)
        #cv2.imshow("left_eye", left)
        
        ##########裁剪左眼图像，可以扩充数据集
        # if len(left)>0:
            # cv2.imwrite(savePath1 + str(Img), left)
        # cv2.imwrite(savePath1 + str(Img), cv2.cvtColor(get_MER(x1, x2, y1, y2, image), cv2.COLOR_BGR2GRAY))

    # 右眼
        rx1 = int(right_eye[0] - W / 2)
        if (rx1 <= 0):
            rx1 = 1
        rx2 = int(right_eye[0] + W / 2)
        if(rx2>=639):
            rx2=638
        ry1 = int(right_eye[1] - H / 2)-5
        if (ry1 <= 0):
            ry1 = 1
        ry2 = int(right_eye[1] + H / 2)
        if (ry2 >= 479):
            ry1 = 478
        right=get_MER(rx1, rx2, ry1, ry2, image)

        label_right_eye=get_label(right)
        if label_right_eye == 'closed_eye':
            right_state='close'
            blink = blink+1
        else:
            right_state='open'
            right_eye_count = right_eye_count + 1
        print(label_right_eye)
        cv2.putText(image, "{}".format(right_state), (rx1, ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1,
                     8)
        cv2.putText(image, "right_eye:{}".format(right_state), (5, 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 0), 1, 8)
        # cv2.imshow("right_eye",right)
               # if len(right)>0:
        cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        # cv2.imwrite(savePath2 + str(Img), cv2.cvtColor(get_MER(rx1, rx2, ry1, ry2, image), cv2.COLOR_BGR2GRAY))
      
        




        D=(mouth_left[1]-nose[1])/cos(arc)-(mouth_left[1]-mouth_right[1])/(2*cos(arc))
        m1=nose[1]+D/2+10
        m2=nose[1]+3*D/2+20
        xm1=int(mouth_left[0])
        xm2=int(mouth_right[0])+10
        if(m2>=479):
            m2=478
        if(xm1<=1):
            xm1=2
        if(xm2>=639):
            xm2=638
        mouth=get_MER(int(mouth_left[0]), int(mouth_right[0]), int(m1), int(m2), image)
        mouth_label = get_label(mouth)
        if mouth_label=='open_mouth':
            mouth_count=mouth_count+1

            mouth_state='open_mouth'
        elif mouth_label=='closed_mouth':
            mouth_state='closed'
            mouth_count=0
            frag = False
        else:
            mouth_state='Eat or Smoke'

        if mouth_count>=10:
            mouth_count=0
            mouth_state = 'Yawd'
            frag=True


        # cv2.imshow("mouth", mouth)
        # if frag:
        #     cv2.putText(image, "State:Yawd", (5, 105),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 100), 2, 8)
        # 耳部，(上面那个模型没有训练打电话这个类别，单独训练了一个耳朵和打电话的二分类模型)
        # left_ear
        #lex1 = x1 - 3 * W
        #if int(lex1)<=10:
        #    lex1=10
        #lex2 = bounding_box[0]
        #ley1 = y1
        #ley2 = bounding_box[1] + bounding_box[3]
        #left_ear=get_MER(int(lex1),int(lex2),int(ley1),int(ley2),image)
        #left_ear_label=get_call_label(left_ear)
        #print(left_ear_label)
        # cv2.imwrite(savePath2 + 'lear'+str(img_count)+'.jpg', left_ear)

        # print(left_ear_label)
        # if left_ear_label=='calling':
        #     ear_state='Calling'
        #     mouth_state='Talking'
        #     cv2.putText(image, "other_state:{}".format(ear_state), (5, 75),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 1, 8)


        # cv2.imshow("left_ear", left_ear)
        # right_ear
        #rex2 =rx2 + 3 * W
        #if int(rex2)>=479:
        #     rex2=478
        #if int(rex2) <= rex1:
        #    rex2 = rex1+30
        #rey1 = y1
        #right_ear = get_MER(int(rex1), int(rex2), int(rey1), int(rey2), image)
        # cv2.imwrite(savePath2 + 'rear' + str(i)+'.jpg', right_ear)
        # i=i+1
        #right_ear_label = get_call_label(right_ear)
        # print(right_ear_label)
        # if right_ear_label=='calling':
        #     ear_state='Calling'
        #     mouth_state='Talking'
        #     cv2.putText(image, "other_state:{}".format(ear_state), (5, 75),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 1, 8)
        # cv2.imshow("right_ear", right_ear)
        # 嘴部状态显示
        cv2.putText(image, "mouth_state:{}".format(mouth_state), (int(mouth_left[0]) - 20, int(m1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 255), 1, 8)
        cv2.putText(image, "mouth_state:{}".format(mouth_state), (5, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, 8)


        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)
        cv2.rectangle(image, (int(mouth_left[0]), int(m1)), (int(mouth_right[0]), int(m2)), (0, 0, 255), 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)
        cv2.rectangle(image, (int(lex1), int(ley1)), (int(lex2), int(ley2)), (0, 255, 0), 1)
        cv2.rectangle(image, (int(rex1), int(rey1)), (int(rex2), int(rey2)), (0, 255, 0), 1)
        # print(mouth_label)
        # gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        image=cv2.resize(image,(640,480))

        # cv2.imwrite(savepath + str(i)+'.jpg', gray)
        # i+=1
        T = time.time() - start
        fps = 1 / T  # 实时在视频上显示fps
        #
        fps_txt = 'fps:%.2f' % (fps)
        cv2.putText(image, fps_txt, (0,180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, 8)
        cv2.imshow("image",image)
        if cv2.waitKey(1)==27:
            cv2.destroyAllWindows()

# print(time.time() - start)
# print("FPS")
# print(len(imagelist) / (time.time()-start))
# print("测试图片数量：{}".format(img_count))
# print("闭嘴图片预测数量{}".format(mouth_count))
# mv=mouth_count/img_count
# print("闭嘴图片测试准确率{}".format(mv))
# print("左眼闭合预测数量{}".format(left_eye_count))
# print("右眼闭合的数量{}".format(right_eye_count))
# lv=left_eye_count/img_count
# print("左眼测试准确率{}".format(lv))
# rv=right_eye_count/img_count
# print("右眼测试准确率{}".format(rv))

