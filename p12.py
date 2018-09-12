import cv2
from darkflow.net.build import TFNet
import numpy as np
import time


option = {
        'model' : 'cfg/yolo.cfg',
        'load':   "C:/darkflow-master/bin/yolov2.weights",
        "threshold" : 0.15,
        'cpu' : 0.8
}



tfnet = TFNet(option)



capture = cv2.VideoCapture("videofile.mp4")
colors = [tuple(255*np.random.rand(3)) for i in range(10)]

while(capture.isOpened( )):
    stime =  time.time()
    ret , frame = capture.read()
    result = tfnet.return_predict(frame)    
    if ret :
        for color , result in zip(colors , result):
            top_left = (result['topleft']['x'] , result['topleft']['y'])
            bottom_right = (result['bottomright']['x'] , result['bottomright']['y']) 
            label = result['label']
            frame = cv2.rectangle(frame , top_left , bottom_right , color , 8 ) 
            frame = cv2.putText(frame , label , top_left ,cv2.FONT_HERSHEY_COMPLEX , 1 , (0,0,0) , 2)
        frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        cv2.imshow('frame' , frame)
        print('FPS {:.1f}'.format(1/(time.time()-stime)))
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
       



'''


tfnet = TFNet(option)
capture = cv2.VideoCapture('videofile.mp4')
colors = [tuple(255*np.random.rand(3)) for i in range(10)]

size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)



i = 0
frame_rate_divider = 3
while(capture.isOpened()):
    ret, frame = capture.read()
    result = tfnet.return_predict(frame)
    if ret:
        if i % frame_rate_divider == 0:
            for color , result in zip(colors , result):
                top_left = (result['topleft']['x'] , result['topleft']['y'])
                bottom_right = (result['bottomright']['x'] , result['bottomright']['y']) 
                label = result['label']
                frame = cv2.rectangle(frame , top_left , bottom_right , color , 8 ) 
                frame = cv2.putText(frame , label , top_left ,cv2.FONT_HERSHEY_COMPLEX , 1 , (0,0,0) , 2)
            frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            cv2.imshow('frame', frame)
            i += 1
        else:
            i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()


'''