import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks
from .interpolation import bilinear





def scale_translate(src: cp.ndarray, scale: float, tx: int, ty: int, dst_width: int | None = None, dst_height: int | None = None) -> cp.ndarray:
    """
    Applies scale and translation to an image.
    Uses bilinear interpolation

    Parameters:
        src: cp.ndarray (H x W x 3), dtype=uint8
        scale: float
        tx: int
        ty: int
        dst_width: int | None = None - if None, uses the same as input
        dst_height: int | None = None - ...

    Returns:
        dst: cp.ndarray (dst_height x dst_width x 3), dtype=uint8
    """

    src_width = src.shape[1]
    src_height = src.shape[0]

    if dst_width is None:
        dst_width = src_width

    if dst_height is None:
        dst_height = src_height

    # Create output array
    dst = cp.empty((dst_height, dst_width, 3), dtype=cp.uint8)

    blocks = make_blocks(dst_width, dst_height)

    # Call kernel
    Kernels.scale_translate(
        blocks, THREADS,
        (
            src.ravel(),
            cp.int32(src_width),
            cp.int32(src_height),
            dst.ravel(),
            cp.int32(dst_width),
            cp.int32(dst_height),
            cp.float32(scale),
            cp.int32(tx),
            cp.int32(ty)
        )
    )
    return dst


def scale(src: cp.ndarray, scale: float, dst_width: int | None = None, dst_height: int | None = None) -> cp.ndarray:
    """
    Applies scaling to an image.
    Uses bilinear interpolation
    (essentially calls `scale_translate` with no translation)

    Parameters:
        src: cp.ndarray (H x W x 3), dtype=uint8
        scale: float
        dst_width: int | None = None - if None, uses the same as input
        dst_height: int | None = None - ...

    Returns:
        dst: cp.ndarray (dst_height x dst_width x 3), dtype=uint8
    """

    return scale_translate(
        src=src,
        scale=scale,
        tx=0,
        ty=0,
        dst_width=dst_width,
        dst_height=dst_height
    )


def resize(src: cp.ndarray, dst_width: int, dst_height: int) -> cp.ndarray:
    """
    Resizes into a destination width and height.
    Applies bilinear interpolation.

    Parameters:
        src: cp.ndarray
        dst_width: int
        dst_height: int

    Returns:
        cp.ndarray
    """

    src_width = src.shape[1]
    src_height = src.shape[0]

    # Create output array
    dst = cp.empty((dst_height, dst_width, 3), dtype=cp.uint8)

    blocks = make_blocks(dst_width, dst_height)

    Kernels.resize(
        blocks, THREADS,
        (
            src.ravel(),
            cp.int32(src_width),
            cp.int32(src_height),
            dst.ravel(),
            cp.int32(dst_width),
            cp.int32(dst_height)
        )
    )
    return dst



def crop(src: cp.ndarray, left: int | float, right: int | float, top: int | float, bottom: int | float) -> cp.ndarray:
    """
    Crop the array in a specific box
    NOTE: This does not use a custom kernel, it just slices the array. We define it this way for abstraction.

    Left and top are included, right and bottom are excluded.

    Coordinates may be floats, this would allow bilinear interpolation

    Parameters:
        src: cp.ndarray
        left: int | float
        right: int | float
        top: int | float
        bottom: int | float

    Returns:
        cp.ndarray
    """

    # Verify that if floats are being passed, they allow forming a box
    diff_x = right - left
    diff_y = bottom - top

    assert int(diff_x) == diff_x and int(diff_y) == diff_y, "If floats are passed, make sure that they form an integer box"

    if left == int(left) and top == int(top):
        # Normal crop
        return src[top:bottom, left:right, :]
    
    else:
        # Requires interpolation of at least one axis
        x_distance = left - int(left)
        y_distance = top - int(top)

        left1 = int(left)
        left2 = left1 + 1

        right1 = int(right)
        right2 = right1 + 1

        top1 = int(top)
        top2 = top1 + 1

        bottom1 = int(bottom)
        bottom2 = bottom1 + 1

        if top != int(top) and left != int(left):
            # Requires interpolation on both axis
            top_left = src[top1:bottom1, left1:right1, :]
            top_right = src[top1:bottom1, left2:right2, :]

            bottom_left = src[top2:bottom2, left1:right1, :]
            bottom_right = src[top2:bottom2, left2:right2, :]

        elif top != int(top):
            # Requires interpolation only on Y axis
            top_left = top_right = src[top1:bottom1, left1:right1, :]
            bottom_left = bottom_right = src[top2:bottom2, left1:right1, :]

        else:
            # Requires interpolation only on X axis
            top_left = bottom_left = src[top1:bottom1, left1:right1, :]
            top_right = bottom_right = src[top1:bottom1, left2:right2, :]


        return bilinear(
            top_left, top_right, bottom_left, bottom_right,
            x_distance, y_distance
        )



def overlay(src: cp.ndarray, overlay: cp.ndarray, x: int, y: int, alpha: float = 1) -> None:
    """
    Overlays an array on the source one, at the given position, with optional custom alpha channel
    Modifies it inplace

    Parameters:
        src: cp.ndarray
        overlay: cp.ndarray
        x: int
        y: int
        alpha: float = 1
    """

    # Calculate coords box
    top_y = y
    bottom_y = top_y+overlay.shape[0]
    left_x = x
    right_x = left_x+overlay.shape[1]

    # Make sure coords aren't bigger than the source
    bottom_y = min(src.shape[0], bottom_y)
    right_x = min(src.shape[1], right_x)

    # Clip overlay to fit in the coords
    overlay_w = right_x - left_x
    overlay_h = bottom_y - top_y

    clipped_overlay = crop(overlay, 0, overlay_w, 0, overlay_h)
    
    if alpha == 1:
        # Overlay is fully opaque, we can simply replace the pixels
        src[top_y:bottom_y, left_x:right_x, :] = clipped_overlay
    else:
        # We have to use a custom kernel
        blocks = make_blocks(overlay_w, overlay_h)

        # Ravel the source and overlay where it needs overlaying
        region = src[top_y:bottom_y, left_x:right_x, :]
        ksrc = region.ravel()
        koverlay = clipped_overlay.ravel()

        Kernels.overlay_opacity(
            blocks, THREADS,
            (
                ksrc,
                koverlay,
                cp.int32(overlay_w),
                cp.int32(overlay_h),
                cp.float32(alpha)
            )
        )

        if not region.flags['C_CONTIGUOUS']:
            # Only assign back if ravel created a copy (non-contiguous slice)
            src[top_y:bottom_y, left_x:right_x, :] = ksrc.reshape((overlay_h, overlay_w, -1))


def blend(src: cp.ndarray, overlay: cp.ndarray, x: int, y: int) -> None:
    """
    Blend an overlay on a frame, with custom alpha channel for each pixel.

    Parameters:
        src: cp.ndarray of 3 channels, RGB
        overlay: cp.ndarray of 4 channels, RGBA
        x: int
        y: int

    Returns:
        None
    """

    # Calculate coords box
    top_y = y
    bottom_y = top_y+overlay.shape[0]
    left_x = x
    right_x = left_x+overlay.shape[1]

    # Make sure coords aren't bigger than the source
    bottom_y = min(src.shape[0], bottom_y)
    right_x = min(src.shape[1], right_x)

    # Clip overlay to fit in the coords
    overlay_w = right_x - left_x
    overlay_h = bottom_y - top_y

    clipped_overlay = crop(overlay, 0, overlay_w, 0, overlay_h)

    blocks = make_blocks(overlay_w, overlay_h)

    # Ravel the source and overlay where it needs overlaying
    region = src[top_y:bottom_y, left_x:right_x, :]
    ksrc = region.ravel()
    koverlay = clipped_overlay.ravel()

    Kernels.blend(
        blocks, THREADS,
        (
            ksrc,
            koverlay,
            cp.int32(overlay_w),
            cp.int32(overlay_h)
        )
    )

    if not region.flags['C_CONTIGUOUS']:
        # Only assign back if ravel created a copy (non-contiguous slice)
        src[top_y:bottom_y, left_x:right_x, :] = ksrc.reshape((overlay_h, overlay_w, -1))
