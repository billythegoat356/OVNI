

THREADS = (16, 16)

def make_blocks(width: int, height: int) -> tuple[int, int]:
    """
    Makes the blocks needed for a CUDA kernel to operate on a specific width and height with a specific amount of threads

    Parameters:
        width: int
        height: int

    Returns:
        tuple[int, int]
    """
    return (
        (width + THREADS[0] - 1) // THREADS[0],
        (height + THREADS[1] - 1) // THREADS[1]
    )
