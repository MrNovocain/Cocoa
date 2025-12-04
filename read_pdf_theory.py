
import pypdf
import sys
import re

def read_pdf(file_path):
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_path = r"w:\Research\NP\Cocoa\CGS Cai.pdf"
    content = read_pdf(pdf_path)
    
    # Keywords to search for theoretical context
    keywords = ["MSE", "bias", "variance", "lambda", "convex combination", "first order", "m1", "m2", "difference"]
    
    print(f"Total characters: {len(content)}")
    
    # Simple keyword context extraction
    for kw in keywords:
        print(f"\n--- Context for '{kw}' ---")
        matches = [m.start() for m in re.finditer(re.escape(kw), content, re.IGNORECASE)]
        for idx in matches[:5]: # Limit to first 5 matches per keyword
            start = max(0, idx - 200)
            end = min(len(content), idx + 300)
            print(f"...{content[start:end].replace(chr(10), ' ')}...")

