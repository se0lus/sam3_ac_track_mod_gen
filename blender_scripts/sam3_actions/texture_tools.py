"""
Texture unpacking, format conversion, and material conversion operators.

Three operators:
1. SAM3_OT_unpack_textures     - Unpack embedded textures from GLB/glTF imports
2. SAM3_OT_convert_textures_png - Convert JPG/JPEG textures to PNG format
3. SAM3_OT_convert_materials_bsdf - Convert all materials to Principled BSDF
"""

from __future__ import annotations

import os

import bpy  # type: ignore[import-not-found]

from . import ActionSpec


# ---------------------------------------------------------------------------
# Helper utilities (pure-Python, testable without Blender)
# ---------------------------------------------------------------------------

def texture_output_path(image_name: str, file_format: str, texture_dir: str) -> str:
    """
    Build the absolute output path for an unpacked texture.

    Parameters
    ----------
    image_name : str
        A sanitised base name for the image (no extension).
    file_format : str
        Blender file_format string, e.g. "PNG", "JPEG".
    texture_dir : str
        The directory where textures will be written.

    Returns
    -------
    str
        Full path including extension.
    """
    ext = format_to_extension(file_format)
    return os.path.join(texture_dir, image_name + ext)


def format_to_extension(file_format: str) -> str:
    """Map a Blender image file_format to a file extension."""
    mapping = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "BMP": ".bmp",
        "TIFF": ".tiff",
        "TARGA": ".tga",
        "OPEN_EXR": ".exr",
        "HDR": ".hdr",
    }
    return mapping.get(file_format.upper(), ".png")


def is_jpeg_path(filepath: str) -> bool:
    """Return True if *filepath* has a JPEG extension."""
    return os.path.splitext(filepath.lower())[1] in (".jpg", ".jpeg")


def jpg_to_png_path(filepath: str) -> str:
    """
    Given a filepath ending in .jpg / .jpeg, return the equivalent .png path.
    If it already ends in .png (or other), return as-is.
    """
    root, ext = os.path.splitext(filepath)
    if ext.lower() in (".jpg", ".jpeg"):
        return root + ".png"
    return filepath


def try_import_pil():
    """Attempt to import PIL. Returns the Image module or None."""
    try:
        from PIL import Image  # type: ignore[import-untyped]
        return Image
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Operator 1: Unpack Textures
# ---------------------------------------------------------------------------

class SAM3_OT_unpack_textures(bpy.types.Operator):
    """Unpack all embedded textures from the current Blender scene."""

    bl_idname = "sam3.unpack_textures"
    bl_label = "Unpack Textures"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        # Ensure a "texture" sub-directory exists next to the .blend file.
        blend_dir = bpy.path.abspath("//")
        if not blend_dir:
            self.report({"ERROR"}, "Save the .blend file first (need a base directory)")
            return {"CANCELLED"}

        texture_dir = os.path.join(blend_dir, "texture")
        os.makedirs(texture_dir, exist_ok=True)

        unpacked_count = 0
        image_counter = 0

        for image in bpy.data.images:
            if not image.packed_file or image.type != "IMAGE":
                continue

            image_name = f"image_{image_counter}"
            image_counter += 1

            out_path = texture_output_path(image_name, image.file_format, texture_dir)

            # Write packed data to disk.
            try:
                with open(out_path, "wb") as f:
                    f.write(image.packed_file.data)
            except Exception as e:
                self.report({"WARNING"}, f"Failed to write {out_path}: {e}")
                continue

            # Remove packed data from .blend and point to the external file.
            image.unpack(method="REMOVE")

            # Use Blender-relative path.
            rel_path = bpy.path.relpath(out_path)
            image.filepath = rel_path
            image.reload()

            print(f"[SAM3] unpack image: {out_path}")
            unpacked_count += 1

        self.report({"INFO"}, f"Unpacked {unpacked_count} texture(s) to {texture_dir}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator 2: Convert Textures to PNG
# ---------------------------------------------------------------------------

class SAM3_OT_convert_textures_png(bpy.types.Operator):
    """Convert all JPG/JPEG textures used by scene materials to PNG."""

    bl_idname = "sam3.convert_textures_png"
    bl_label = "Convert Textures to PNG"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        PILImage = try_import_pil()
        converted = 0

        for image in bpy.data.images:
            if image.packed_file or image.type != "IMAGE":
                continue

            filepath = bpy.path.abspath(image.filepath)
            if not is_jpeg_path(filepath):
                continue

            png_path = jpg_to_png_path(filepath)

            # Strategy 1: Use PIL if available.
            if PILImage is not None:
                try:
                    pil_img = PILImage.open(filepath)
                    pil_img.save(png_path)
                except Exception as e:
                    self.report({"WARNING"}, f"PIL conversion failed for {filepath}: {e}")
                    continue
            else:
                # Strategy 2: Use Blender's save_render as fallback.
                try:
                    old_format = image.file_format
                    image.file_format = "PNG"
                    image.save_render(png_path)
                    image.file_format = old_format
                except Exception as e:
                    self.report({"WARNING"}, f"Blender conversion failed for {filepath}: {e}")
                    continue

            # Update image reference to point to the new PNG.
            image.filepath = bpy.path.relpath(png_path)
            image.reload()
            print(f"[SAM3] converted to PNG: {png_path}")
            converted += 1

        method = "PIL" if PILImage is not None else "Blender save_render"
        self.report({"INFO"}, f"Converted {converted} texture(s) to PNG (method: {method})")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator 3: Convert Materials to Principled BSDF
# ---------------------------------------------------------------------------

class SAM3_OT_convert_materials_bsdf(bpy.types.Operator):
    """Convert all materials to Principled BSDF, preserving texture assignments."""

    bl_idname = "sam3.convert_materials_bsdf"
    bl_label = "Convert Materials to Principled BSDF"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        converted = 0

        for material in bpy.data.materials:
            if not material.node_tree:
                continue

            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # Skip materials that already have Principled BSDF connected to output.
            existing_bsdf = None
            output_node = None
            for node in nodes:
                if node.type == "BSDF_PRINCIPLED":
                    existing_bsdf = node
                if node.type == "OUTPUT_MATERIAL":
                    output_node = node

            if existing_bsdf is not None and output_node is not None:
                # Check if BSDF is already connected to output.
                already_connected = False
                for link in links:
                    if link.from_node == existing_bsdf and link.to_node == output_node:
                        already_connected = True
                        break
                if already_connected:
                    # Already properly connected, skip this material.
                    continue
                # BSDF exists but not connected -- will reconnect below.

            # Find the texture image node (look for any TEX_IMAGE node).
            tex_image_node = None
            for node in nodes:
                if node.type == "TEX_IMAGE":
                    tex_image_node = node
                    break

            if output_node is None:
                continue

            # Find existing shader node connected to output (Emission, etc.)
            source_shader = None
            for link in links:
                if link.to_node == output_node:
                    source_shader = link.from_node
                    break

            # If the source is already Principled BSDF, skip.
            if source_shader is not None and source_shader.type == "BSDF_PRINCIPLED":
                continue

            # If there's no texture, nothing meaningful to convert.
            if tex_image_node is None:
                continue

            # Create or get Principled BSDF node.
            bsdf = existing_bsdf
            if bsdf is None:
                bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

            # Link texture -> BSDF Base Color.
            links.new(tex_image_node.outputs[0], bsdf.inputs[0])

            # Remove old connection to material output.
            for link in list(links):
                if link.to_node == output_node:
                    links.remove(link)

            # Link BSDF -> Material Output.
            links.new(bsdf.outputs[0], output_node.inputs[0])

            print(f"[SAM3] convert to BSDF: {material.name}")
            converted += 1

        self.report({"INFO"}, f"Converted {converted} material(s) to Principled BSDF")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Action registration
# ---------------------------------------------------------------------------

ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_unpack_textures,
        menu_label="Unpack Textures",
        icon="PACKAGE",
    ),
    ActionSpec(
        operator_cls=SAM3_OT_convert_textures_png,
        menu_label="Convert Textures to PNG",
        icon="IMAGE_DATA",
    ),
    ActionSpec(
        operator_cls=SAM3_OT_convert_materials_bsdf,
        menu_label="Convert Materials to Principled BSDF",
        icon="NODE_MATERIAL",
    ),
]
