"""
test the python code the get the screenshot of given window title
"""

from utils.win import get_hWnd_by_name , get_screenshot_by_hwnd
import win32con
if __name__ == '__main__':
    keywords = r'MuMu'
    hWnd = get_hWnd_by_name(keywords)
    img = get_screenshot_by_hwnd(hWnd)
    img.show()

    
