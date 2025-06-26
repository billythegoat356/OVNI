import PyNvVideoCodec as nvc



# Constants (adjust code if you need customzz)
GPU_ID = 0
CODEC = 'h264' # Encoding codec
BITRATE = '5M'
PRESET = 'P3' # 1-7, determines quality, but impacts speed


# Set Warning log level (not sure if this works)
nvc.logger.setLevel(nvc.logging.WARNING)