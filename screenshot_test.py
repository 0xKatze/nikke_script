"""
test the python code the get the screenshot of given window title
"""

from utils.win import get_hWnd_by_name , get_screenshot_by_hwnd
import win32con
from PIL import Image
import numpy as np


if __name__ == '__main__':
    keywords = r'Lark'
    hWnd = get_hWnd_by_name(keywords)
    numpy_image = get_screenshot_by_hwnd(hWnd,0,1) # []
    PIL_image = Image.fromarray(np.uint8(numpy_image[:,:,::-1]))
    PIL_image.show()
    
