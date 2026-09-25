import mujoco
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("model/cyberrunner.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

show = False
step = 0

def create_display(m, d):
    plt.ion()
    fig, ax = plt.subplots()
    
    # Instead of recreating the plot every time, declare it once and update it in the loop.
    renderer.update_scene(data, camera="fov_camera")
    arr = renderer.render()
    ax.set_axis_off()
    image = ax.imshow(arr)

    return fig, image, renderer

def display(image, fig, arr):
    # Pass in and update with new image array
    image.set_data(arr)
    # Render data into the internal buffer
    fig.canvas.draw()
    # Pause while loop to update GUI with changes in buffer
    fig.canvas.flush_events()

if show:
    fig, image, renderer = create_display(model, data)

while True:
    if step >= 100:
        print(data.time)
        step = 0
    step += 1

    # Dynamically read ctrlrange parameter values from specified in xml
    low = model.actuator_ctrlrange[:, 0] # -4000
    high = model.actuator_ctrlrange[:, 1] # 4000
    
    # Apply random control values within each actuator's allowed range.
    data.ctrl[:] = np.random.uniform(low, high, size=model.nu)
    
    mujoco.mj_step(model, data, nstep=16)

    renderer.update_scene(data, camera="fov_camera")
    arr = renderer.render()
    if show:
        display(image, fig, arr)

    
    
