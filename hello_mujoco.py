import mujoco
import os
import time
import imageio.v3 as iio
from PIL import Image
import numpy as np
from pathlib import Path


# Load model, every some fps, save image to a dir
m = mujoco.MjModel.from_xml_path("/home/maxma/projects/hello_mujoco/model/cyberrunner.xml")
d = mujoco.MjData(m)
# Looks at current state and draws it as an image
r = mujoco.Renderer(m)

fps = float(30)
frame_num = 0
capture = 1.0 / fps


output_dir= Path("captured_frames")
os.makedirs(output_dir, exist_ok=True)

while d.time < 60:

    low = m.actuator_ctrlrange[:, 0]
    high = m.actuator_ctrlrange[:, 1]
    d.ctrl[:] = np.random.uniform(low, high, size=m.nu)

    # Advance sim by one time step, defined in the xml
    mujoco.mj_step(m, d)
    
    time.sleep(capture)
    r.update_scene(d, camera="fov_camera")
    arr = r.render()
    image = Image.fromarray(arr)
    path = output_dir / f"frame{frame_num}.png"
    iio.imwrite(path, image)
    frame_num += 1

r.close()


    



    



