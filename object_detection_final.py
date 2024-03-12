import cv2
import matplotlib.pyplot as plt
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'
model=cv2.dnn_DetectionModel(frozen_model,config_file)
classLabels=[]
file_name='Labels.txt'
with open(file_name,'rt')as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')
print(classLabels)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

#image

img=cv2.imread('test image.jpg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
ClassIndex,confidece,bbox=model.detect(img,confThreshold=0.3)
font_scale=1
font=cv2.FONT_HERSHEY_PLAIN
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,255,100),thickness=1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

#video or webcam

cap=cv2.VideoCapture('people.mp4')

#checking video opens correctly
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("CANNOT OPEN VIDEO")
font_scale=2
font=cv2.FONT_HERSHEY_COMPLEX

while True:
    ret,frame=cap.read()
    ClassIndex,confidece,bbox=model.detect(frame,confThreshold=0.2)
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(0,0,0),3)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+people.mp40,boxes[1]+40),font,fontScale=font_scale,color=(255,255,255),thickness=2)
    cv2.imshow('OBJECT DETECTION TEST',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()