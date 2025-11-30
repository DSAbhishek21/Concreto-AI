# encode.py — Run this ONCE to generate the base64 string
import base64

with open("background.jpg", "rb") as img_file:
    b64 = base64.b64encode(img_file.read()).decode()

print(f"data:image/jpeg;base64,{b64}")