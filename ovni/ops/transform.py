import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks
from .interpolation import bilinear_blend, linear_blend





def scale_translate(src: cp.ndarray, scale: float, tx: float, ty: float, dst_width: int | None = None, dst_height: int | None = None) -> cp.ndarray:
    """
    Applies scale and translation to an image.
    Uses bilinear interpolation

    Parameters:
        src: cp.ndarray (H x W x 3), dtype=uint8
        scale: float
        tx: float
        ty: float
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
            cp.float32(tx),
            cp.float32(ty)
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

def translate(src: cp.ndarray, tx: float, ty: float, dst_width: int | None = None, dst_height: int | None = None) -> cp.ndarray:
    """
    Applies translation to an image.
    Uses bilinear interpolation
    (essentially calls `scale_translate` with no scale)

    Parameters:
        src: cp.ndarray (H x W x 3), dtype=uint8
        tx: float
        ty: float
        dst_width: int | None = None - if None, uses the same as input
        dst_height: int | None = None - ...

    Returns:
        dst: cp.ndarray (dst_height x dst_width x 3), dtype=uint8
    """

    return scale_translate(
        src=src,
        scale=1,
        tx=tx,
        ty=ty,
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
    width = right - left
    height = bottom - top

    assert abs(round(width)-width) < 10e-5 and abs(round(height)-height) < 10e-5, "If floats are passed, make sure that they form an integer box"

    if left == int(left) and top == int(top):
        # Normal crop if integers are passed
        return src[top:bottom, left:right, :]
    
    else:
        # Otherwise use translation
        return translate(src, -left, -top, round(width), round(height))



def overlay(src: cp.ndarray, overlay_arr: cp.ndarray, x: int | float, y: int | float, alpha: float = 1) -> None:
    """
    Overlays an array on the source one, at the given position, with optional custom alpha channel
    Non integer position is allowed, in this case the image will be interpolated.
    Modifies it inplace
    ---------
    NOTE
    The support for blending isn't fully optimized. We overlay it 2/4 times and blend, then again on the final frame.
    We could instead rewrite the cuda kernel to blend the pixels directly.

    Parameters:
        src: cp.ndarray
        overlay_arr: cp.ndarray
        x: int | float
        y: int | float
        alpha: float = 1
    """

    width = src.shape[1]
    height = src.shape[0]

    # Calculate coords box - from where to where the overlay should be put
    top_y = int(y)
    bottom_y = top_y+overlay_arr.shape[0]
    left_x = int(x)
    right_x = left_x+overlay_arr.shape[1]

    # Make sure coords aren't bigger than the source
    right_x = min(width, right_x)
    bottom_y = min(height, bottom_y)

    # Clip overlay to fit in the coords
    overlay_w = right_x - left_x
    overlay_h = bottom_y - top_y
    clipped_overlay = crop(overlay_arr, 0, overlay_w, 0, overlay_h)

    if x != int(x) or y != int(y):
        # Floats passed, requires interpolation
        x_distance = x - int(x)
        y_distance = y - int(y)

        # Clip the source for the area containing the 4 positions of the overlay
        clipped_src = crop(src, left_x, min(width, right_x+1), top_y, min(height, bottom_y+1))

        if x != int(x) and y != int(y):
            # Requires interpolation on both axis
            top_left = clipped_src.copy()
            top_right = clipped_src.copy()

            bottom_left = clipped_src.copy()
            bottom_right = clipped_src.copy()

            overlay(top_left, clipped_overlay, 0, 0)
            overlay(top_right, clipped_overlay, 1, 0)

            overlay(bottom_left, clipped_overlay, 0, 1)
            overlay(bottom_right, clipped_overlay, 1, 1)

            arr = bilinear_blend(
                top_left, top_right, bottom_left, bottom_right,
                x_distance, y_distance
            )


        elif y != int(y):
            # Requires interpolation only on Y axis
            top = clipped_src.copy()
            bottom = clipped_src.copy()

            overlay(top, clipped_overlay, 0, 0)
            overlay(bottom, clipped_overlay, 0, 1)

            arr = linear_blend(
                top, bottom,
                y_distance
            )

        else:
            # Requires interpolation only on X axis
            left = clipped_src.copy()
            right = clipped_src.copy()

            overlay(left, clipped_overlay, 0, 0)
            overlay(right, clipped_overlay, 1, 0)

            arr = linear_blend(
                left, right,
                x_distance
            )

        overlay(src, arr, left_x, top_y, alpha=alpha) # Only apply alpha once

    else:
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


def blend(src: cp.ndarray, overlay_arr: cp.ndarray, x: int | float, y: int | float) -> None:
    """
    Blend an overlay on a frame, with custom alpha channel for each pixel.
    Non integer position is allowed, in this case the image will be interpolated.
    Modifies it in place

    Parameters:
        src: cp.ndarray of 3 channels, RGB
        overlay_arr: cp.ndarray of 4 channels, RGBA
        x: int | float
        y: int | float

    Returns:
        None
    """

    width = src.shape[1]
    height = src.shape[0]

    # Calculate coords box - from where to where the overlay should be put
    top_y = int(y)
    bottom_y = top_y+overlay_arr.shape[0]
    left_x = int(x)
    right_x = left_x+overlay_arr.shape[1]

    # Make sure coords aren't bigger than the source
    right_x = min(width, right_x)
    bottom_y = min(height, bottom_y)

    # Clip overlay to fit in the coords
    overlay_w = right_x - left_x
    overlay_h = bottom_y - top_y
    clipped_overlay = crop(overlay_arr, 0, overlay_w, 0, overlay_h)

    if x != int(x) or y != int(y):
        # Floats passed, requires interpolation
        x_distance = x - int(x)
        y_distance = y - int(y)

        # Clip the source for the area containing the 4 positions of the overlay
        clipped_src = crop(src, left_x, min(width, right_x+1), top_y, min(height, bottom_y+1))

        if x != int(x) and y != int(y):
            # Requires interpolation on both axis
            top_left = clipped_src.copy()
            top_right = clipped_src.copy()

            bottom_left = clipped_src.copy()
            bottom_right = clipped_src.copy()

            blend(top_left, clipped_overlay, 0, 0)
            blend(top_right, clipped_overlay, 1, 0)

            blend(bottom_left, clipped_overlay, 0, 1)
            blend(bottom_right, clipped_overlay, 1, 1)

            arr = bilinear_blend(
                top_left, top_right, bottom_left, bottom_right,
                x_distance, y_distance
            )

        elif y != int(y):
            # Requires interpolation only on Y axis
            top = clipped_src.copy()
            bottom = clipped_src.copy()

            blend(top, clipped_overlay, 0, 0)
            blend(bottom, clipped_overlay, 0, 1)

            arr = linear_blend(
                top, bottom,
                y_distance
            )

        else:
            # Requires interpolation only on X axis
            left = clipped_src.copy()
            right = clipped_src.copy()

            blend(left, clipped_overlay, 0, 0)
            blend(right, clipped_overlay, 1, 0)

            arr = linear_blend(
                left, left,
                x_distance
            )

        
        overlay(src, arr, left_x, top_y) # No alpha now

    else:
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
