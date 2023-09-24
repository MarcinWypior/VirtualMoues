import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import math

def find_distance():
    x1, y1 = lmList[12][1], lmList[12][2]
    x2, y2 = lmList[8][1], lmList[8][2]

    length = math.hypot(x2 - x1, y2 - y1)

    return length



################################
wCam, hCam = 640,480
frameR = 100 # Frame Reduction
smoothening =5
################################

pTime = 0
plocX,plocY = 0,0
clocX,clocY =0,0


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(3,480)
detector = htm.handDetector(maxHands=1)
wScr ,hScr = autopy.screen.size()
#print(wScr,hScr)
#1536.0 864.0

while True:
    #1. Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    # 2. Get the tip of the index and middle finger
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        #print(x1,x2,y1,y2)

        #3. Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)


        #4. Only Index Finger : Moving Mode
        if fingers[1]==1 and fingers[2]==0:
            #5. Convert Coordinates

            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            #6. Smoothen Values
            clocX = plocX + (x3-plocX)/smoothening
            clocY = plocY + (y3-plocY)/smoothening


            #7. Move Mouse
            autopy.mouse.move(wScr- clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX,plocY = clocX,clocY

        #8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1]==1 and fingers[2]==1:
            # 9. Find distance between fingers
            length = find_distance()
            print(length)
            if length <45:
                #10. Click Mouse if distance short
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()



        #11. Frame Rate
        #12. Display

    cTime = time.time()
    fps = 1 /(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)

    cv2.imshow("image", img)

    cv2.waitKey(1)


