import mujoco
import cv2
import numpy as np

model = mujoco.MjModel.from_xml_path("model/cyberrunner.xml")
data = mujoco.MjData(model)
#renderer = mujoco.Renderer(model, height=720, width=1280)
# Scale down using same width to height ratio
renderer = mujoco.Renderer(model, height=360, width=640)

display = True

step = 0
while True:
    if step >= 100:
        print(data.time)
        step = 0
    step += 1

    low = model.actuator_ctrlrange[:, 0] # -4000
    high = model.actuator_ctrlrange[:, 1] # 4000
    data.ctrl[:] = np.random.uniform(low, high, size=model.nu)

    mujoco.mj_step(model, data, nstep=36)

    renderer.update_scene(data, camera="fov_camera")
    pixels = renderer.render()

    if display:
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("MuJoCo Camera View", bgr_pixels)
        cv2.pollKey()