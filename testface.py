import cv2
#   对应类别标签
Name = {'1':'liuyi','2':'zy','3':'pengyuyan','4':'zzx'}
# 加载训练数据集
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./facemodel/trainer02.yml')

# 准备识别的图片
im = cv2.imread('./allmag/03/zx12.jpg')
grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# 检测人脸
face_detector = cv2.CascadeClassifier('./tools/haarcascade_frontalface_default.xml')
face = face_detector.detectMultiScale(grey)

for x, y, w, h in face:
    # 返回人脸标签和可信度，可信度数值越低，可信度越高（用词不当，不要在意）
    label, confidence = recognizer.predict(grey[y:y+h, x:x+w])
# 这里将60作为界限，当检测检测值为60时，我们就确定人物
    if confidence <= 60:
        # 绘制预测框
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        Label = Name[str(label)]
        # 显示标签
        cv2.putText(im,Label,(x, y),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 225), 2)

    else:
        print("未匹配到数据")
    im = cv2.resize(im, (800, 600))
    cv2.imshow('im', im)
    cv2.waitKey(0)
# 销毁窗口
cv2.destroyAllWindows()

