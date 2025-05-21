import cv2
import numpy as np
import time
import mediapipe as mp
from pynput.keyboard import Controller, Key as PynputKey

# HandTracker class using Mediapipe
class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    def isPeaceSign(self, lmList):
        """
        Detects a "peace sign" gesture:
        - Index and middle fingers up
        - Ring, pinky, and thumb down
        """
        if not lmList:
            return False

        # Finger tip IDs in Mediapipe
        tipIds = [4, 8, 12, 16, 20]

        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Check for peace sign: index and middle fingers up, others down
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            return True
        return False

# Key class representing each virtual key
class Key:
    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text

    def drawKey(self, img, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        # Draw semi-transparent rectangle
        bg_rec = img[self.y : self.y + self.h, self.x : self.x + self.w]
        overlay = bg_rec.copy()
        overlay[:] = bg_color
        cv2.addWeighted(overlay, alpha, bg_rec, 1 - alpha, 0, bg_rec)

        # Put the letter/text
        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y), fontFace, fontScale, text_color, thickness)

    def isOver(self, x, y):
        return self.x < x < self.x + self.w and self.y < y < self.y + self.h

# Helper functions
def getMousePos(event, x, y, flags, param):
    global clickedX, clickedY
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONUP:
        clickedX, clickedY = x, y
    if event == cv2.EVENT_MOUSEMOVE:
        mouseX, mouseY = x, y

def calculateDistance(pt1, pt2):
    return int(((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5)

# Initialize keyboard controller
keyboard = Controller()

# Initialize HandTracker
tracker = HandTracker(detectionCon=0.8, trackCon=0.8)

# Initialize video capture
cap = cv2.VideoCapture(0)
ptime = 0

# Define virtual keyboard layout
w, h = 80, 60
startX, startY = 40, 200
keys = []
letters = list("QWERTYUIOPASDFGHJKLZXCVBNM")
for i, l in enumerate(letters):
    if i < 10:
        keys.append(Key(startX + i * (w + 5), startY, w, h, l))
    elif i < 19:
        keys.append(Key(startX + (i - 10) * (w + 5), startY + h + 5, w, h, l))
    else:
        keys.append(Key(startX + (i - 19) * (w + 5), startY + 2 * (h + 10), w, h, l))

# Add special keys
keys.append(Key(startX, startY + 3 * (h + 10), 5 * (w + 5), h, "Space"))
keys.append(Key(startX + 6 * (w + 5), startY + 3 * (h + 10), 5 * (w + 5), h, "<--"))
keys.append(Key(startX + 12 * (w + 5), startY + 3 * (h + 10), 5 * (w + 5), h, "Enter"))

# Additional control keys
showKey = Key(40, 50, 80, 50, 'Show')
exitKey = Key(40, 120, 80, 50, 'Exit')
textBox = Key(startX, startY - h - 10, 10 * (w + 5), h, '')

# Adjust window size and position
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame from camera.")
    exit()
frameHeight, frameWidth, _ = frame.shape
showKey.x = frameWidth - 100
exitKey.x = frameWidth - 100
textBox.x = startX
textBox.y = startY - h - 10

# Initialize variables
clickedX, clickedY = 0, 0
mouseX, mouseY = 0, 0
show = False
cv2.namedWindow('video')
cv2.setMouseCallback('video', getMousePos)
counter = 0
previousClick = 0

# Main loop
while True:
    if counter > 0:
        counter -= 1

    signTipX = 0
    signTipY = 0
    thumbTipX = 0
    thumbTipY = 0

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror image

    # Find hands
    frame = tracker.findHands(frame, draw=True)
    lmList = tracker.getPosition(frame, draw=False)

    if lmList:
        # Get tip positions
        signTip = lmList[8]   # Index finger tip
        thumbTip = lmList[4]  # Thumb tip
        signTipX, signTipY = signTip[1], signTip[2]
        thumbTipX, thumbTipY = thumbTip[1], thumbTip[2]

        # Calculate distance between index finger tip and thumb tip
        distance = calculateDistance((signTipX, signTipY), (thumbTipX, thumbTipY))

        # If distance is small, consider it a "click"
        if distance < 40:
            currentTime = time.time()
            if currentTime - previousClick > 0.3:
                clickedX, clickedY = signTipX, signTipY
                previousClick = currentTime

        # Check for gesture recognition
        if tracker.isPeaceSign(lmList):
            # Trigger the copy action (Ctrl+C)
            keyboard.press(PynputKey.ctrl_l)
            keyboard.press('c')
            keyboard.release('c')
            keyboard.release(PynputKey.ctrl_l)
    
    # Draw the keyboard
    for key in keys:
        key.drawKey(frame)
    
    # Draw the special keys
    showKey.drawKey(frame)
    exitKey.drawKey(frame)
    textBox.drawKey(frame)

    # Update textBox
    if show:
        textBox.text = 'Showing Keyboard'
    else:
        textBox.text = ''

    # Check if clicked position is within any key
    if clickedX and clickedY:
        for key in keys:
            if key.isOver(clickedX, clickedY):
                if key.text == 'Space':
                    keyboard.press(' ')
                    keyboard.release(' ')
                elif key.text == '<--':
                    keyboard.press(PynputKey.backspace)
                    keyboard.release(PynputKey.backspace)
                elif key.text == 'Enter':
                    keyboard.press(PynputKey.enter)
                    keyboard.release(PynputKey.enter)
                else:
                    keyboard.press(key.text)
                    keyboard.release(key.text)
                clickedX, clickedY = 0, 0
                break
        if showKey.isOver(clickedX, clickedY):
            show = not show
            clickedX, clickedY = 0, 0
        if exitKey.isOver(clickedX, clickedY):
            break
    
    # Display the frame
    cv2.imshow('video', frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
