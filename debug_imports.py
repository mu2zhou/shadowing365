try:
    import PIL
    print(f"PIL version: {PIL.__version__}")
    from PIL import Image, ImageDraw, ImageFont
    print("PIL submodules imported.")
except ImportError as e:
    print(f"PIL Import Error: {e}")

try:
    import moviepy
    print(f"MoviePy version: {moviepy.__version__}")
    from moviepy.editor import AudioFileClip
    print("MoviePy submodules imported.")
except ImportError as e:
    print(f"MoviePy Import Error: {e}")

try:
    import numpy
    print(f"Numpy version: {numpy.__version__}")
except ImportError as e:
    print(f"Numpy Import Error: {e}")
