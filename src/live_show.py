import cv2
from pathlib import Path
import time

frames_dir = Path("live_frames")

shown = set()

FPS = 30
frame_delay = 1 / FPS

print("🎥 Live viewer avviato")

cv2.namedWindow("LIVE", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LIVE", 1280, 720)  

last_frame = None

import pygame

pygame.mixer.init()
pygame.mixer.music.load("input.wav")

audio_started = False

last_display_time = 0
DISPLAY_INTERVAL = 8.0  # secondi

while True:

    

    images = sorted(frames_dir.glob("*.png"))

    new_images = [p for p in images if p not in shown]

    if new_images and not audio_started:
        print("🔊 Start audio")
        pygame.mixer.music.play()
        audio_started = True

    if new_images:
        for img_path in new_images:

            now = time.time()

            # aspetta almeno 2 secondi tra un'immagine e l'altra
            if now - last_display_time < DISPLAY_INTERVAL:
                continue

            img = cv2.imread(str(img_path))

            if img is None:
                continue

            cv2.imshow("LIVE", img)

            shown.add(img_path)
            last_frame = img
            last_display_time = now  # aggiorna tempo

            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                exit()

    else:
        # se non ci sono nuove immagini, continua a mostrare l'ultima
        if last_frame is not None:
            cv2.imshow("LIVE", last_frame)
            cv2.waitKey(1)

        time.sleep(0.01)