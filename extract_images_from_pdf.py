import os
import fitz  # PyMuPDF
from PIL import Image, ImageFilter
import io

def apply_median_filter(image, size=3):
    """
    Apply a median filter to the given image.
    :param image: PIL Image object
    :param size: Filter size (default is 3)
    :return: Filtered PIL Image object
    """
    return image.filter(ImageFilter.MedianFilter(size))

def extract_images_from_pdf(pdf_folder):
    output_folder = os.path.join(pdf_folder, "extracted_images")
    filtered_folder = os.path.join(pdf_folder, "filtered_images")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(filtered_folder, exist_ok=True)

    # Process each PDF in the folder
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc):
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]  # Extract image reference
                    image = doc.extract_image(xref)
                    img_bytes = image["image"]

                    # Convert to PIL image
                    img_pil = Image.open(io.BytesIO(img_bytes))

                    # Save unfiltered image
                    unfiltered_filename = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}_page{page_num+1}.png")
                    img_pil.save(unfiltered_filename, "PNG")

                    # Apply median filter and save filtered image
                    img_filtered = apply_median_filter(img_pil)
                    filtered_filename = os.path.join(filtered_folder, f"{os.path.splitext(pdf_file)[0]}_page{page_num+1}_filtered.png")
                    img_filtered.save(filtered_filename, "PNG")

                    print(shape.img_pil)
                    print(f"Unfiltered image saved: {unfiltered_filename}")
                    print(f"Filtered image saved: {filtered_filename}")

    print("✅ Image extraction and filtering complete!")

# Set the folder containing PDFs
pdf_folder = "test_images_1"

extract_images_from_pdf(pdf_folder)
