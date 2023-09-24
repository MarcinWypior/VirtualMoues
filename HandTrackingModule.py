import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands=2,detectionCon= 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode= self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img,handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myhand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (0,0, 255), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[3][1] < self.lmList[4][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img,draw=True)
        lmList = detector.findPosition(img,draw=True)
        if len(lmList) != 0:
            print(lmList[4])

        print("len ", len(lmList))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        img = cv2.flip(img,1)
        cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)


        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()