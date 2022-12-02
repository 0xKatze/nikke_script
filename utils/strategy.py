"""
    strategy for auto weak point aiming
"""

import numpy as np
import win32gui
from utils import mouse
from utils.yolov5.yolov5_onnx import YOLOV5_ONNX
from utils.win import get_screenshot_by_hwnd

# match aim_box to mouse

class Simulator(object):
    def __init__(self, hWnd, config):
        self.top_hWnd = hWnd
        self.config = config
        self.detector = YOLOV5_ONNX(config.onnx_path)

    def screenshot(self):
        return get_screenshot_by_hwnd(self.top_hWnd,0,1)

    def move_cur_center(self):
        rect = win32gui.GetWindowRect(self.top_hWnd)
        center = [int(rect[2]-rect[0])//2, int(rect[3]-rect[1])//2]
        self.move_to(np.array(center),0)
        self.mouse_point = center
        self.center = center

    def left_down(self):
        mouse.left_down(self.top_hWnd, self.mouse_point[0], self.mouse_point[1])

    def move_to(self,target,lbutton=1):
        if lbutton:
            mouse.mouse_drag(self.top_hWnd,self.mouse_point[0],self.mouse_point[1],target[0],target[1],self.config.is_smooth,self.config.duration,self.config.smooth_k)
        else:
            mouse.move_to(self.top_hWnd,target[0],target[1],0)

    def fix_aim_offset(self):
        # first move the mouse to center of the simulator and then press
        self.move_cur_center()
        self.left_down()
        x = None
        while True:
            img = self.screenshot()
            print('screenshot img_size:', img.shape)
            results = self.detector.infer(img)
            det = results[0]
            if det is not None and len(det):
                for *xywh,conf,cls in det:
                    if int(cls) == 1: # find aim_box
                        x = xywh
                        break
            if not x is None:
                break
        self._offset = np.array(x)[0:2] - self.center

    def aim_alert(self):
        aim_box_center = self.mouse_point + self._offset
        img = self.screenshot()
        results = self.detector.infer(img)

        # find nearest alert
        x , min_dis = None, 1e9
        det = results[0]
        if det is not None and len(det):
            for *xywh, conf, cls in det:
                c = np.array(xywh)[0:2]
                if (int(cls)) == 0:
                    dis = np.linalg.norm(c - aim_box_center)
                    if dis < min_dis:
                        min_dis = dis
                        x = c
        if x is None:
            return 0
        else:
            print('检测到弱点，位置 : ', x)
            offset = x - aim_box_center
            target_mouse_point = self.mouse_point + offset
            self.move_to(target_mouse_point)

    def exit_battle(self):
        """
            not implement yet
        """
        pass

    def turnoff_auto_aiming(self):
        """
            not implement yet
        """
        pass

    def start_simulation(self):
        # first we assume the auto aiming program is off
        self.turnoff_auto_aiming()
        print('开始寻找准心...')
        self.fix_aim_offset()
        print("准心调整成功,开始启动自动瞄准，请勿点击游戏")
        while True:
            self.aim_alert()


class MuMuX(Simulator):
    def __init__(self, hWnd, config):
        super().__init__(hWnd, config)
        # find childWnd
        def callback(hWnd,lParam):
            length = win32gui.GetWindowTextLength(hWnd)
            if (length == 0):
                return True
            windowTitle = win32gui.GetWindowText(hWnd)
            callback._hWndList.append(hWnd)

            return True
        callback._hWndList = []
        win32gui.EnumChildWindows(hWnd,callback,None)
        self.top_hWnd = callback._hWndList[0]

