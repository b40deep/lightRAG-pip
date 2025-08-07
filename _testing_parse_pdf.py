import os
WORKING_DIR = os.path.join(os.getcwd(),  "t2d-ground-truth")

def use_pypdf():
    from pypdf import PdfReader
    # creating a pdf reader object
    # reader = PdfReader(os.path.join(WORKING_DIR, 'Intellectuals.pdf'))
    # for page in reader.pages:
    #     print(page.extract_text())
    for pdf_file in os.listdir(WORKING_DIR):
        if pdf_file.endswith(".pdf"):
            all_pages = ""
            reader = PdfReader(os.path.join(WORKING_DIR, pdf_file))
            for page in reader.pages:
                all_pages += " " + str(page.extract_text())
            print(f"Extracted text from {pdf_file}:\n{all_pages}\n")
            print("=" * 40)
            print("=" * 40)
            print("\n")
        
def use_marker_pdf():
    import pypdfium2 # Needs to be at the top to avoid warnings
    import argparse
    import os
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.converters import convert_single_pdf
    from marker.logger import configure_logging
    from marker.models import load_all_models
    from marker.output import save_markdown

    # converter = PdfConverter(
    #     artifact_dict=create_model_dict(),
    # )
    # rendered = converter(os.path.join(WORKING_DIR, 'Management-of-type-2-Diabetes-Electronic-2014.pdf'))
    # text, _, images = text_from_rendered(rendered)
    model_lst = load_all_models()
    full_text, images, out_meta = convert_single_pdf(fname, model_lst)
    fname = os.path.basename(fname)
    subfolder_path = save_markdown('marker-output', fname, full_text, images, out_meta)
    print(f"Saved markdown to the {subfolder_path} folder")
    print(f"Extracted text from :\n{full_text}\n")
    print("=" * 40)
    print("\n")

def parse_md():
    import markdown
    folders = [folder.path for folder in os.scandir(WORKING_DIR) if folder.is_dir()]
    for folder in folders:
        for md_file in os.listdir(folder):
            if md_file.endswith(".md"):
                md_text = ""
                full_path = os.path.join(folder,md_file)
                print(f"Processing {md_file}...\n from {full_path}")
                with open(full_path, 'r', encoding="utf8") as f:
                    text = f.read()
                    md_text = markdown.markdown(text)
                print(f"Extracted text from {md_file}:\n{md_text}\n")
                print("=" * 40)
                print("=" * 40)
                print("\n")


def main():
    # use_pypdf()
    # use_marker_pdf()
    parse_md()
    print("PDF text extraction completed.")

if __name__ == "__main__":
    main()
    print("\nDone!")