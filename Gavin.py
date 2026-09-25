import mujoco
import mujoco.viewer
import random
import time
import cv2

model = mujoco.MjModel.from_xml_path("model/cyberrunner.xml")

renderer = mujoco.Renderer(model, height=360, width=640)
time = 0
skip = 0
fps = 55
x = 0
y = 0

data = mujoco.MjData(model)
while True:
    #print(data.qpos)
    if (skip >= 3):
        skip = 0
        x = random.randrange(-3000, 3000)
        y = random.randrange(-3000, 3000)
    data.actuator('tilt_x').ctrl = x
    data.actuator('tilt_y').ctrl = y
    time = data.time
    skip += 1
    for _ in range(16):
        mujoco.mj_step(model, data)

    renderer.update_scene(data, camera="fov_camera")
    pixels = renderer.render()

    bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    cv2.imshow("MuJoCo Camera View", bgr_pixels)
    cv2.waitKey(1)
