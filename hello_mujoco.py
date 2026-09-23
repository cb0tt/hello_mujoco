import mujoco

import mujoco.viewer

import time



model = mujoco.MjModel.from_xml_path("/home/maxma/projects/hello_mujoco/model/cyberrunner.xml")

# Current state of the model while the simulation runs.
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, width=640, height=480)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # Advance simulation, takes parameters model and simulation state
        mujoco.mj_step(model, data)

        #  Updates the displayed scene with latest MjModel & MjData
        viewer.sync()


