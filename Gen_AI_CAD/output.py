import cadquery as cq
import math

class Gear:
    def __init__(self, num_teeth=12, pressure=0.1, width=0.05, height=0.02):
        self.num_teeth = num_teeth
        self.pressure = pressure
        self.width = width
        self.height = height
        self.gear = cq.Workplane("XY")

        # Calculate the gear's radius
        gear_radius = width / num_teeth

        # Create the gear profile
        gear_profile = cq.Gear(num_teeth, gear_radius)

        # Extrude the gear profile to create the gear body
        gear_body = gear_profile.extrude(height)

        # Apply pressure to the gear body
        gear_body = gear_body.pressure(pressure)

        # Rotate the gear body by 90 degrees clockwise
        gear_body = gear_body.rotate((0, 0, 1), 90, 0)

        # Translate the gear body to the desired position
        gear_body = gear_body.translate((0, 0, 0))

        # Add the gear body to the CAD assembly
        self.gear = gear_body

    def get_sketch(self):
        return self.gear.val

    def get_parameters(self):
        return self.pressure, self.width, self.height, self.num_teeth

# Create a gear with 12 teeth and a pressure of 0.1
gear = Gear(num_teeth=12, pressure=0.1, width=0.05, height=0.02)

# Export the gear to a STEP file
cq.exporters.step(gear.get_sketch(), 'gear.step')