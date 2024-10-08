import time
from pynput import mouse

# 存储鼠标移动事件和时间戳
mouse_events = []

# 临时变量,记录上一条
tX = None
tY = None
tS = None

# 结束点
pointX = None
pointY = None

# 起始点
lastX = 0
lastY = 0

# 每个移动轨迹起始时间戳
lastStap = 0
# 是否是第一条记录
startFlag = True
# 每个移动轨迹起始时间戳
lastTime = 0


class PathWeaverListener(mouse.Listener):
    def __init__(self):
        global tS, lastX, lastY
        tS = time.time()
        super().__init__(on_move=self.on_move, on_click=self.on_click)

    def on_move(self, x, y):
        global tX, tY, tS, lastStap, lastTime, startFlag
        now = time.time()

        if now - lastStap > 0.3:
            global tS, pointX, pointY
            tS = now
            pointX = x
            pointY = y
            process_mouse_events()
            startFlag = True
        lastStap = now

        if startFlag:
            lastTime = time.time()
            startFlag = False

        if tX != x or tY != y:
            tX = x
            tY = y
            a = time.time()
            mouse_events.append((a - tS, x, y))
            tS = a

    def on_click(self, x, y, button, pressed):
        pass
        # 可以在这里处理鼠标点击事件
        # if pressed:
        #     global tS, pointX, pointY
        #     tS = time.time()
        #     pointX = x
        #     pointY = y
        #     process_mouse_events()


def process_mouse_events():
    if len(mouse_events) == 0:
        return

    global lastX, lastY
    with open("mouse_events.txt", "a") as f:
        index = 1
        for event in mouse_events:
            f.write(f"{index}:{event[0]:.7f}:{event[1]}:{event[2]}\n")
            index += 1
        f.write(f"P{lastX}:{lastY}:{pointX}:{pointY}:{lastStap - lastTime:.7f}:{len(mouse_events)}\n")
        print(f"Write{lastX}:{lastY}:{pointX}:{pointY}:{lastStap - lastTime:.7f}:{len(mouse_events)}")
        lastX = pointX
        lastY = pointY
        mouse_events.clear()


if __name__ == "__main__":
    listener = PathWeaverListener()
    listener.start()

    try:
        # 阻塞主线程，直到用户按下 Enter 键
        input("Press Enter to stop...")
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        listener.join()  # 确保监听器线程完全停止
