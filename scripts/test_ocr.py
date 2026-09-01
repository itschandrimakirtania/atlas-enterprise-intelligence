from PIL import Image, ImageDraw
import pytesseract

# Tell pytesseract where the Tesseract executable is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Create a simple white image
image = Image.new("RGB", (800, 250), "white")

# Put text onto the image
draw = ImageDraw.Draw(image)
draw.text(
    (50, 100),
    "Atlas Enterprise Intelligence",
    fill="black",
)

# Ask Tesseract to recognize the text
extracted_text = pytesseract.image_to_string(image)

print("OCR result:")
print(extracted_text)