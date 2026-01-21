from PIL import Image, ImageDraw, ImageFont
import os

def test_font_rendering():
    # Configured path
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    output_image = "debug_font_test.png"
    
    print(f"Testing font: {font_path}")
    if not os.path.exists(font_path):
        print("ERROR: Font file does not exist!")
        return

    try:
        # Try loading with index 0
        font = ImageFont.truetype(font_path, 40) # removed index=0 for broad compat first
    except Exception as e:
        print(f"ERROR: Failed to load font: {e}")
        return

    img = Image.new('RGB', (800, 400), color='#fdf6e3')
    draw = ImageDraw.Draw(img)
    
    text = "English: Hello\nChinese: 你好世界\nGerman: Hallo Welt"
    
    try:
        draw.text((50, 50), text, font=font, fill='#073642')
        print("Text drawn.")
        img.save(output_image)
        print(f"Saved test image to {output_image}")
    except Exception as e:
        print(f"ERROR: Failed to draw text: {e}")

if __name__ == "__main__":
    test_font_rendering()
