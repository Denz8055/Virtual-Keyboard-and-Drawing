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
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Check for peace sign: index and middle fingers up, others down
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            return True
        return False

    def isFist(self, lmList):
        """
        Detects a "fist" gesture:
        - All fingers (including thumb) are closed
        """
        if not lmList:
            return False

        # Finger tip IDs in Mediapipe
        tipIds = [4, 8, 12, 16, 20]

        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] > lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Check for fist: all fingers closed
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
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
w, h = 50, 50  # Adjusted key size
startX, startY = 20, 100  # Adjusted start position
keys = []
letters = list("QWERTYUIOPASDFGHJKLZXCVBNM")

# Adjust keyboard layout to fit within the video frame
for i, l in enumerate(letters):
    row = i // 10
    col = i % 10
    keys.append(Key(startX + col * (w + 5), startY + row * (h + 5), w, h, l))

# Add special keys
keys.append(Key(startX, startY + 4 * (h + 5), 6 * (w + 5), h, "Space"))
keys.append(Key(startX + 7 * (w + 5), startY + 4 * (h + 5), 4 * (w + 5), h, "<--"))
keys.append(Key(startX + 12 * (w + 5), startY + 4 * (h + 5), 6 * (w + 5), h, "Enter"))

# Additional control keys
controlKeyWidth = 100
controlKeyHeight = 50
showKey = Key(startX, startY - controlKeyHeight - 20, controlKeyWidth, controlKeyHeight, 'Show')
exitKey = Key(startX + controlKeyWidth + 10, startY - controlKeyHeight - 20, controlKeyWidth, controlKeyHeight, 'Exit')
textBox = Key(startX, startY - h - 10, 10 * (w + 5), h, '')

# Initialize variables
clickedX, clickedY = 0, 0
mouseX, mouseY = 0, 0
show = False
cv2.namedWindow('video', cv2.WINDOW_NORMAL)  # Use resizable window
cv2.resizeWindow('video', 1280, 720)  # Adjust the window size
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
            keyboard.press(PynputKey.ctrl)
            keyboard.press('c')
            keyboard.release('c')
            keyboard.release(PynputKey.ctrl)
            cv2.putText(frame, "Copy Action Triggered", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            time.sleep(0.5)  # Prevent multiple triggers

        if tracker.isFist(lmList):
            # Trigger the paste action (Ctrl+V)
            keyboard.press(PynputKey.ctrl)
            keyboard.press('v')
            keyboard.release('v')
            keyboard.release(PynputKey.ctrl)
            cv2.putText(frame, "Paste Action Triggered", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            time.sleep(0.5)  # Prevent multiple triggers

    # Draw FPS
    ctime = time.time()
    fps = int(1 / (ctime - ptime)) if ctime - ptime > 0 else 0
    ptime = ctime
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw control keys
    showKey.drawKey(frame, (255, 255, 255), (50, 50, 50), alpha=0.6, fontScale=0.6, thickness=2)
    exitKey.drawKey(frame, (255, 255, 255), (50, 50, 50), alpha=0.6, fontScale=0.6, thickness=2)

    if show:
        # Draw text box
        textBox.drawKey(frame, (255, 255, 255), (0, 0, 0), alpha=0.6, fontScale=0.6, thickness=2)

        # Draw all keys
        for key in keys:
            # Determine if cursor or finger is over the key
            if key.isOver(mouseX, mouseY) or key.isOver(signTipX, signTipY):
                key_alpha = 0.3
                if key.isOver(clickedX, clickedY):
                    # Simulate key press
                    if key.text == "<--":
                        textBox.text = textBox.text[:-1]
                        keyboard.press(PynputKey.backspace)
                        keyboard.release(PynputKey.backspace)
                    elif key.text == "Space":
                        textBox.text += " "
                        keyboard.press(PynputKey.space)
                        keyboard.release(PynputKey.space)
                    elif key.text == "Enter":
                        textBox.text += "\n"
                        keyboard.press(PynputKey.enter)
                        keyboard.release(PynputKey.enter)
                    else:
                        textBox.text += key.text
                        keyboard.press(key.text.lower())
                        keyboard.release(key.text.lower())
                    clickedX, clickedY = 0, 0  # Reset click
            else:
                key_alpha = 0.6

            key.drawKey(frame, alpha=key_alpha, fontScale=0.8, thickness=2)

    # Handle control keys
    if showKey.isOver(clickedX, clickedY):
        show = not show
        showKey.text = "Hide" if show else "Show"
        clickedX, clickedY = 0, 0

    if exitKey.isOver(clickedX, clickedY):
        break

    # Display the frame
    cv2.imshow('video', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

