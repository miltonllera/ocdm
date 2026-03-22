import bpy
import os
import math
import colorsys
import itertools
import csv

# =========================
# TOP-LEVEL CONSTANTS
# =========================
OUTPUT_DIR = "data/temp/pentominos_3d"
CAMERA = "Camera"

NUM_COLORS = 6
NUM_ANGLES = 8              # object rotation around Y axis
ROT_MIN = 0.0
ROT_MAX = 2 * math.pi

NUM_CAM_ANGLES = 5          # camera pivot around origin
CAM_MIN = -math.pi / 6
CAM_MAX = math.pi / 6


# =========================
# PARAMETER GENERATION
# =========================
def linspace(start, end, num, endpoint):
    if num == 1:
        return [(start - end) / 2]
    step = (end - start) / (num - (1 if endpoint else 0))
    return [start + i * step for i in range(num)]


def generate_colors(n):
    hues = linspace(0.0, 1.0, n, False)
    return [(h, (*colorsys.hsv_to_rgb(h, 1.0, 1.0), 1.0)) for h in hues]


# =========================
# HELPER FUNCTIONS
# =========================
def set_object_to_origin(obj):
    obj.location = (0, 0, 0)  # raised slightly above the ground


def set_rotation(obj, angle):
    obj.rotation_euler = (angle, 0, 0)


def set_color(obj, rgba):
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="Material")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]

    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba


def rotate_camera_around_origin(cam, angle, base_radius, base_z):
    cam.location.x = base_radius * math.cos(angle)
    cam.location.y = base_radius * math.sin(angle)
    cam.location.z = base_z

    direction = -cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()


def reset_object(obj, original_state):
    obj.location = original_state["location"]
    obj.rotation_euler = original_state["rotation"]


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")

    scene = bpy.context.scene

    # Set resolution to 256x256
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256

    collections = bpy.data.collections  # type: ignore
    camera = bpy.data.objects[CAMERA]  # type: ignore
    pentominos = [obj for obj in collections['Pentominos'].objects if obj.type == 'MESH']
    bg_objects = sorted(
        [obj for obj in collections['Background'].objects if obj.type == 'MESH'],
        key=lambda o: o.name
    )

    rotations = linspace(ROT_MIN, ROT_MAX, NUM_ANGLES, False)
    camera_angles = linspace(CAM_MIN, CAM_MAX, NUM_CAM_ANGLES, True)
    colors = generate_colors(NUM_COLORS)

    original_states = {}
    for obj in pentominos:
        original_states[obj.name] = {
            "location": obj.location.copy(),
            "rotation": obj.rotation_euler.copy(),
        }

    base_radius = camera.location.length
    base_z = camera.location.z

    bg_color_columns = [f"{obj.name}_color_hue" for obj in bg_objects]

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "filename",
            "object",
            "rotation_y",
            "color_hue",
            *bg_color_columns,
            "camera_angle"
        ])

        counter = 0

        for obj in pentominos:
            for params in itertools.product(rotations, colors, *([colors] * len(bg_objects)), camera_angles):
                rot = params[0]
                color_hue, color_rgba = params[1]
                bg_colors = [(h, rgba) for h, rgba in params[2:2 + len(bg_objects)]]
                cam_angle = params[-1]

                reset_object(obj, original_states[obj.name])

                set_object_to_origin(obj)
                set_rotation(obj, rot)
                set_color(obj, color_rgba)

                for bg_obj, (_, bg_rgba) in zip(bg_objects, bg_colors):
                    set_color(bg_obj, bg_rgba)

                rotate_camera_around_origin(camera, cam_angle, base_radius, base_z)

                filename = f"{obj.name}_{counter:07d}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                bpy.context.scene.render.filepath = filepath

                bpy.ops.render.render(write_still=True)

                writer.writerow([
                    filename,
                    obj.name,
                    rot,
                    color_hue,
                    *[h for h, _ in bg_colors],
                    cam_angle
                ])

                counter += 1

            reset_object(obj, original_states[obj.name])

    print(f"Done! Rendered {counter} images.")
    print(f"Metadata saved to: {csv_path}")


main()
