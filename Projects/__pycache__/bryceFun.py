# ---------------------------------------
import cv2
from pynput import keyboard
from USBController import USBController
from PWMController import PWMController
# ----------------------------------------

usb = USBController()
throttle = PWMController(0, usb)
turn = PWMController(1, usb)

def on_press(key):
    try:
        key = key.char
    except:
        key = key.name
    # Forward
    if key == "w":
        print("in here")
        throttle.set(.4)
    # Backward
    if key == "s":
        throttle.set(-.4)
    # Right
    if key == "d":
        turn.set(.4)
    # Left
    if key == "a":
        turn.set(-.4)
    # Stop
    if key == "x":
        throttle.reset()
        turn.reset()
        exit()


listen = keyboard.Listener(on_press=on_press)
listen.start()
listen.join()

throttle.reset()
turn.reset()
