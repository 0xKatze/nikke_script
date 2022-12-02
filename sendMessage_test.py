"""
test send Meesage to Simulate APP(MuMu)
"""

# get cmd authorize: https://www.zhihu.com/question/68342805
# https://www.yiibai.com/powershell/powershell-run-as-administrator.html


from utils.win import get_hWnd_by_name,setForeground
from utils.mouse import mouse_click, mouse_drag, __left_down,__move_to,__left_up
import win32gui
import win32api
import win32con

import time


if __name__ == '__main__':
    print(win32con.HTCLIENT)
    print(win32con.WM_SETCURSOR)

    keywords = r'MuMu'
    hWnd = get_hWnd_by_name(keywords)
    print(hWnd)
    def callback(hWnd,lParam):
        length = win32gui.GetWindowTextLength(hWnd)
        if (length == 0):
            return True
        windowTitle = win32gui.GetWindowText(hWnd)
        callback._titleList.append(windowTitle)
        callback._hWndList.append(hWnd)

        return True
    callback._titleList = []
    callback._hWndList = []
    win32gui.EnumChildWindows(hWnd,callback,None)
    print(callback._hWndList)
    childhWnd = callback._hWndList[0]

    #__left_down(hWnd,300,300)
    __left_down(childhWnd,300,300)
    for i in range(4):
        mouse_drag(childhWnd,300,300,700,300,1,1,100)
        mouse_drag(childhWnd,700,300,700,700,1,1,100)
        mouse_drag(childhWnd,700,700,300,700,1,1,100)
        mouse_drag(childhWnd,300,700,300,300,1,1,100)
    __left_up(childhWnd,300,300)
    #win32gui.SendMessage(childhWnd,win32con.WM_CAPTURECHANGED,0,0)
    #__move_to(hWnd,500,300)
    

    #left_click(hWnd,1384-315,225-9)
    
    #mouse_drag()
    """
    
    while True:
        x,y = win32api.GetCursorPos()
        print(x,y)
    """
    