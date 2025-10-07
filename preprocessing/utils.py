from OCC.Core.STEPControl import STEPControl_Reader

# Load Step File
def load_step(path):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(path)
    step_reader.TransferRoots()
    pieze = step_reader.OneShape()

    if status != 1:
        raise Exception(f"Error reading STEP file: {path}")