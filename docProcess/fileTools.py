import base64


def convert_pdf_to_base64(pdf_path: str):
    # Read the PDF file in binary mode, encode it to base64, and decode to string
    with open(pdf_path, "rb") as file:
        base64_encoded_pdf = base64.b64encode(file.read()).decode()
    return base64_encoded_pdf