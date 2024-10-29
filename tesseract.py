from PIL import Image
from pytesseract import pytesseract
import enum

class OS(enum.Enum):
    Mac = 0
    Windows = 1

class Lang(enum.Enum):
    POR = 'por'
    ENG = 'eng'
    RUS = 'rus'
    ITA = 'ita'
    SPA = 'spa'

class ImageReader:

    def __init__(self, os:OS):
        if os == OS.Mac:
            print("Running on: MAC \n")

        if os == OS.Windows:
            windows_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            pytesseract.tesseract_cmd = windows_path
            print("Running on: Windows \n")


    def extract_text(self, image: str, lang: str) -> str:
        with Image.open(image) as img:
            extracted_text = pytesseract.image_to_string(img, lang=lang)
            return extracted_text
        
if __name__ == '__main__':
    image_reader = ImageReader(OS.Windows)
    text = image_reader.extract_text('assets/varias_ling.png', lang = 'spa+por+eng')
    print(text)