import docx2txt

def convert_docx_to_txt(docx_path, txt_path):
    text = docx2txt.process(docx_path)
    with open(txt_path, "w") as txt_file:
        txt_file.write(text)

if __name__ == "__main__":
    docx_path = "C:/Users/Shiva/Desktop/Data structure and Algorithm in Python.docx"  # Replace with your input DOCX file path
    txt_path = 'C:/Users/Shiva/Desktop/mytext.py'  # Replace with your desired output text file path
    convert_docx_to_txt(docx_path, txt_path)




