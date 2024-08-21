import pyautogui
import time

T = 0.5

time.sleep(T)

pyautogui.click(x=200, y=200)

pyautogui.hotkey("ctrl", "b")
time.sleep(T)
pyautogui.hotkey("d")
time.sleep(T)
pyautogui.hotkey("ctrl", "c")
pyautogui.hotkey("enter")
pyautogui.typewrite("tmux kill-server")
time.sleep(T)
pyautogui.hotkey("enter")
time.sleep(T)


pyautogui.typewrite("tmux new -s core")
pyautogui.hotkey("enter")
time.sleep(T)
pyautogui.typewrite("roscore")
pyautogui.hotkey("enter")
time.sleep(T)
pyautogui.hotkey("ctrl", "b")
time.sleep(T)
pyautogui.hotkey("d")

for session in ("robots", "cameras", "computer_vision", "plots"):
    pyautogui.typewrite(f"tmux new -s {session}")
    pyautogui.hotkey("enter")
    time.sleep(T)
    pyautogui.typewrite(f"roslaunch airo_butler {session}.launch")
    pyautogui.hotkey("enter")
    time.sleep(T)
    pyautogui.hotkey("ctrl", "b")
    time.sleep(T)
    pyautogui.hotkey("d")
