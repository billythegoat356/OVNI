import PyNvVideoCodec as nvc



# Constants (defaults for parameters)
GPU_ID = 0
CODEC = 'h264' # Encoding codec
BITRATE = '5M'
PRESET = 'P3' # 1-7, determines quality, but impacts speed


# Set Warning log level (not 100% sure if this works)
nvc.logger.setLevel(nvc.logging.WARNING)