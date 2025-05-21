import cv2
import numpy as np
import time
import mediapipe as mp
import pyautogui
import subprocess
from pynput.mouse import Controller as MouseController, Button as MouseButton
from pynput.keyboard import Controller as KeyboardController, Key

# Initialize HandTracker using Mediapipe
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

    def isFist(self, lmList):
        """Detects a fist gesture (All fingers closed)."""
        if not lmList:
            return False
        tipIds = [4, 8, 12, 16, 20]
        fingers = [(lmList[tipIds[i]][2] > lmList[tipIds[i] - 2][2]) for i in range(1, 5)]
        return all(fingers)

# Initialize controllers
mouse = MouseController()
keyboard = KeyboardController()
tracker = HandTracker(detectionCon=0.8, trackCon=0.8)

# Capture video
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()  # Get screen resolution
ptime = 0

# Variables for gesture-based click and typing
previousClickTime = 0
previousKeyPressTime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror image

    # Find hands
    frame = tracker.findHands(frame, draw=True)
    lmList = tracker.getPosition(frame, draw=False)

    if lmList:
        # Get fingertip positions
        index_finger = lmList[8]  # Index finger tip
        thumb = lmList[4]  # Thumb tip
        middle_finger = lmList[12]  # Middle finger tip

        ix, iy = index_finger[1], index_finger[2]
        tx, ty = thumb[1], thumb[2]
        mx, my = middle_finger[1], middle_finger[2]

        # Map hand coordinates to screen size
        screenX = np.interp(ix, [0, frame.shape[1]], [0, screen_width])
        screenY = np.interp(iy, [0, frame.shape[0]], [0, screen_height])

        # Move mouse to mapped position
        mouse.position = (screenX, screenY)

        # Click if the distance between index and thumb is small
        distance = np.sqrt((ix - tx) ** 2 + (iy - ty) ** 2)
        if distance < 30:  # Click threshold
            currentTime = time.time()
            if currentTime - previousClickTime > 0.3:  # Prevent double clicks
                mouse.click(MouseButton.left)
                previousClickTime = currentTime

        # Right-click if middle finger and thumb touch
        right_click_distance = np.sqrt((mx - tx) ** 2 + (my - ty) ** 2)
        if right_click_distance < 30:
            mouse.click(MouseButton.right)

        # Scroll if index and middle finger move up/down
        if abs(iy - my) > 50:
            if iy < my:  # Scroll up
                pyautogui.scroll(5)
            else:  # Scroll down
                pyautogui.scroll(-5)

        # Open Task Manager if Fist Gesture Detected
        if tracker.isFist(lmList):
            print("Opening Task Manager...")
            subprocess.run("taskmgr", shell=True)
            time.sleep(1)  # Prevent multiple triggers

        # Gesture-based Keyboard Inputs
        currentTime = time.time()
        if currentTime - previousKeyPressTime > 0.5:
            # Victory sign (index + middle extended) = Ctrl+C (Copy)
            if lmList[8][2] < lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] > lmList[14][2]:
                keyboard.press(Key.ctrl)
                keyboard.press('c')
                keyboard.release('c')
                keyboard.release(Key.ctrl)
                print("Gesture: Copy")
                previousKeyPressTime = currentTime

            # Thumbs up = Ctrl+V (Paste)
            if lmList[4][1] > lmList[3][1] and lmList[8][2] > lmList[6][2]:
                keyboard.press(Key.ctrl)
                keyboard.press('v')
                keyboard.release('v')
                keyboard.release(Key.ctrl)
                print("Gesture: Paste")
                previousKeyPressTime = currentTime

            # Pointing Gesture (index extended, others down) = Press Enter
            if lmList[8][2] < lmList[6][2] and all(lmList[i][2] > lmList[i - 2][2] for i in [12, 16, 20]):
                keyboard.press(Key.enter)
                keyboard.release(Key.enter)
                print("Gesture: Enter")
                previousKeyPressTime = currentTime

            # Three fingers up (index + middle + ring) = Open Notepad
            if all(lmList[i][2] < lmList[i - 2][2] for i in [8, 12, 16]) and lmList[20][2] > lmList[18][2]:
                print("Opening Notepad...")
                subprocess.run("notepad", shell=True)
                time.sleep(1)

    # Display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
    ptime = ctime
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Hand Tracking Mouse & Keyboard Control", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
