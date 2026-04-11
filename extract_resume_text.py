from pathlib import Path
from sys import exit
path = Path(r'C:\Users\USER\Desktop\cv\Komiljon_Mukhammadiev_0322.pdf')
print('EXISTS', path.exists())
try:
    import PyPDF2
except ImportError:
    print('NO_PYPDF2')
    exit(0)
reader = PyPDF2.PdfReader(str(path))
for i, page in enumerate(reader.pages, 1):
    print(f'---PAGE {i}---')
    text = page.extract_text()
    print(text)
