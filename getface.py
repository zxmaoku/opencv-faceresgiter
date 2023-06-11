import cv2
import os
import numpy
from imutils import paths
import random

root_path= "./allmag"

def getfaces():###获取图片特征和标签
    global root_path

    ##  获取人脸检测器
    face_dector = cv2.CascadeClassifier("./tools/haarcascade_frontalface_default.xml")
    global root_path
    faces = []
    lables = []
    ##获取图片路径
    imagepath=sorted(list(paths.list_images(root_path)))
    print(imagepath)
    for path in imagepath:
        path=path.replace('\\','/')
        ## 读取图片
        # im = cv2.imread(os.path.join(path))
        im = cv2.imread(path)
        ###  转换灰度
        grey=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ##读取人脸数据  获得人脸位置信息
        face= face_dector.detectMultiScale(grey)
        for x,y,w,h in face:
            label = path.split("/")[-2]
            print(label)
            lables.append(int(label))
            ##将图像分割获得人脸数据
            face_img = grey[y:y+h,x:x+w]
            ##使用INTER_LINEAR方法进行图像缩放
            face_img = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_AREA)
            faces.append(face_img)
    return faces,lables

##调用方法回去人脸信息及标签
faces, labels = getfaces()

##获取训练对象
recognizer =  cv2.face.LBPHFaceRecognizer_create()

##  训练数据
recognizer.train(faces, numpy.array(labels))
recognizer.write('./facemodel/trainer02.yml')