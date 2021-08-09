from PIL import Image
import pytesseract
print(pytesseract.image_to_string(Image.open(
    "C:\\Users\\one1e\\Desktop\\practice\\number.png"), lang="kor", config="--psm 7"))
