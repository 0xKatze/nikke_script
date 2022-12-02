import win32gui
import numpy as np

def get_hWnd_by_name(name):
    def callback(hWnd,lParam):
        
        if not win32gui.IsWindowVisible(hWnd):
            return True
        
        length = win32gui.GetWindowTextLength(hWnd)
        if (length == 0):
            return True
        windowTitle = win32gui.GetWindowText(hWnd)
        callback._titleList.append(windowTitle)

        return True
    callback._titleList = []
    win32gui.EnumWindows(callback,None)
    titleList = callback._titleList
    
    hWnd = None
    for t in titleList:
        if name in t:
            hWnd = win32gui.FindWindow(0,t)
            print(t)
    if hWnd is None:
        win32gui.MessageBox(0,"your app with keyword '{}' not found".format(name),'Error',0)
        return 0

    return hWnd  

import win32gui
import win32api
import win32com.client
import win32con

def setForeground(hWnd):
    if hWnd != win32gui.GetForegroundWindow():
        # why call SendMessage: https://blog.csdn.net/weixin_30299539/article/details/96321161
        win32gui.SendMessage(hWnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(hWnd)
    return 1

from PIL import ImageGrab
def winShot(hWnd):
    rect = win32gui.GetWindowRect(hWnd)
    img = ImageGrab.grab(rect) # [RGB]
    return img

def get_screenshot_by_hwnd(hWnd,call_front=False,is_backgroud=False):
    if not is_backgroud:
        if call_front and setForeground(hWnd):
            img = winShot(hWnd)
            img = np.asarray(img)
            return img
        else:
            return None
    else:
        # to do
        # inference: https://stackoverflow.com/questions/19695214/screenshot-of-inactive-window-printwindow-win32gui
        win32gui.MessageBox(0,'get backgroud app screenshot not implement yet','Warning',0)
        return None


