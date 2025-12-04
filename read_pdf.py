
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
    # Print the first 2000 characters and some keywords context
    print(content[:2000])
    
    keywords = ["convex combination", "weight", "gamma", "post-break", "MFV", "validation"]
    print("\n--- Keyword Search ---")
    for kw in keywords:
        if kw.lower() in content.lower():
            print(f"Found '{kw}':")
            # Find occurrences and print context
            lower_content = content.lower()
            start = 0
            while True:
                idx = lower_content.find(kw.lower(), start)
                if idx == -1:
                    break
                print(f"...{content[max(0, idx-50):min(len(content), idx+100)]}...")
                start = idx + 1
                if start > len(content): break
                # Limit to first 3 occurrences per keyword to avoid spam
                if start > lower_content.find(kw.lower(), lower_content.find(kw.lower()) + 1) + 1:
                     pass 

