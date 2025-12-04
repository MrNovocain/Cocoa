
import pypdf
import sys

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
    
    with open("cgs_paper_content.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved PDF content to cgs_paper_content.txt")
