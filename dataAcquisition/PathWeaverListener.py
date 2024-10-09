import random
import time
from pynput import mouse

# Store mouse movement events and timestamps
mouse_events = []

# Temporary variable, records the data of the previous path point
tX = None
tY = None
tS = None

# start point
lastX = 0
lastY = 0

# end point
pointX = None
pointY = None

# Starting timestamp of each movement trajectory
lastStap = 0
# Is it the first record
startFlag = True
# Starting timestamp of each movement trajectory
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
        # You can handle mouse click events here

def process_mouse_events():
    global lastX, lastY
    if len(mouse_events) == 0:
        lastX = pointX
        lastY = pointY
        return
    with open("mouse_events.txt", "a") as f:
        index = 1
        for event in mouse_events:
            if event[0] == 0:  # Prevent zero delay
                new_event = (random.uniform(0.00099, 0.00245), event[1], event[2])
                f.write(f"{index}:{new_event[0]:.5f}:{new_event[1]}:{new_event[2]}\n")
            else:
                f.write(f"{index}:{event[0]:.5f}:{event[1]}:{event[2]}\n")
            index += 1
        f.write(f"P{lastX}:{lastY}:{pointX}:{pointY}:{lastStap - lastTime:.5f}:{len(mouse_events)}\n")
        print(f"Write{lastX}:{lastY}:{pointX}:{pointY}:{lastStap - lastTime:.5f}:{len(mouse_events)}")
        lastX = pointX
        lastY = pointY
        mouse_events.clear()


if __name__ == "__main__":
    listener = PathWeaverListener()
    listener.start()

    try:
        # Block the main thread until the user presses the Enter key
        input("Press Enter to stop...\n")
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        listener.join()  # Ensure that the listener thread is completely stopped
