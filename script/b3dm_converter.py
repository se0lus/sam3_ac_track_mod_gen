"""
B3DM to GLB converter module.

Converts Batched 3D Model (.b3dm) files to Binary glTF (.glb) format
by stripping the B3DM header and table data to extract the embedded GLB payload.

B3DM format (28-byte header):
  Bytes  0- 3: Magic "b3dm"
  Bytes  4- 7: Version (uint32 LE)
  Bytes  8-11: Total byte length (uint32 LE)
  Bytes 12-15: Feature table JSON byte length (uint32 LE)
  Bytes 16-19: Feature table binary byte length (uint32 LE)
  Bytes 20-23: Batch table JSON byte length (uint32 LE)
  Bytes 24-27: Batch table binary byte length (uint32 LE)
  Remaining:   Feature Table JSON + Binary + Batch Table JSON + Binary + GLB payload
"""

import os
import struct
import logging

logger = logging.getLogger(__name__)

B3DM_MAGIC = b'b3dm'
GLB_MAGIC = b'glTF'
B3DM_HEADER_SIZE = 28


class B3dmConversionError(Exception):
    """Raised when a B3DM file cannot be converted."""
    pass


def convert_file(b3dm_path: str, output_path: str) -> str:
    """Convert a single B3DM file to GLB format.

    Args:
        b3dm_path: Path to the input .b3dm file.
        output_path: Path for the output .glb file.

    Returns:
        The output_path on success.

    Raises:
        FileNotFoundError: If the input file does not exist.
        B3dmConversionError: If the file is not a valid B3DM or contains no GLB payload.
    """
    if not os.path.isfile(b3dm_path):
        raise FileNotFoundError(f"B3DM file not found: {b3dm_path}")

    with open(b3dm_path, 'rb') as f:
        header = f.read(B3DM_HEADER_SIZE)

    if len(header) < B3DM_HEADER_SIZE:
        raise B3dmConversionError(
            f"File too small to be a valid B3DM ({len(header)} bytes): {b3dm_path}"
        )

    magic = header[0:4]
    if magic != B3DM_MAGIC:
        raise B3dmConversionError(
            f"Invalid B3DM magic bytes (got {magic!r}, expected {B3DM_MAGIC!r}): {b3dm_path}"
        )

    # Parse header fields (all uint32 little-endian)
    version, total_length, ft_json_len, ft_bin_len, bt_json_len, bt_bin_len = \
        struct.unpack('<6I', header[4:28])

    glb_offset = B3DM_HEADER_SIZE + ft_json_len + ft_bin_len + bt_json_len + bt_bin_len

    file_size = os.path.getsize(b3dm_path)
    if glb_offset >= file_size:
        raise B3dmConversionError(
            f"No GLB payload found (offset {glb_offset} >= file size {file_size}): {b3dm_path}"
        )

    # Read the GLB payload
    with open(b3dm_path, 'rb') as f:
        f.seek(glb_offset)
        glb_data = f.read()

    # Validate GLB magic
    if len(glb_data) < 4 or glb_data[0:4] != GLB_MAGIC:
        raise B3dmConversionError(
            f"GLB payload does not start with glTF magic bytes: {b3dm_path}"
        )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(glb_data)

    logger.info("Converted %s -> %s (%d bytes)", b3dm_path, output_path, len(glb_data))
    return output_path


def convert_directory(input_dir: str, output_dir: str) -> list:
    """Batch convert all .b3dm files in a directory (recursively) to GLB.

    Args:
        input_dir: Root directory containing .b3dm files.
        output_dir: Root directory for output .glb files (mirrors input structure).

    Returns:
        List of (b3dm_path, glb_path) tuples for successfully converted files.

    Raises:
        FileNotFoundError: If the input directory does not exist.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    converted = []
    errors = []

    for root, _dirs, files in os.walk(input_dir):
        for name in files:
            if not name.lower().endswith('.b3dm'):
                continue

            b3dm_path = os.path.join(root, name)
            # Preserve subdirectory structure
            rel_path = os.path.relpath(b3dm_path, input_dir)
            glb_name = os.path.splitext(rel_path)[0] + '.glb'
            glb_path = os.path.join(output_dir, glb_name)

            try:
                convert_file(b3dm_path, glb_path)
                converted.append((b3dm_path, glb_path))
            except (B3dmConversionError, FileNotFoundError) as e:
                logger.error("Failed to convert %s: %s", b3dm_path, e)
                errors.append((b3dm_path, str(e)))

    logger.info(
        "Batch conversion complete: %d converted, %d errors",
        len(converted), len(errors)
    )
    return converted
