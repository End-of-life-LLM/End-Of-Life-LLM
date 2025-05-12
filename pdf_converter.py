import os
from PyPDF2 import PdfReader

class PdfToTextConverter:
    def __init__(self, input_folder=None, output_folder=None):
        """
        Initialize the converter with optional input and output folders.
        
        Args:
            input_folder (str, optional): Directory containing PDF files to convert
            output_folder (str, optional): Directory to save text files
        """
        self.input_folder = input_folder
        self.output_folder = output_folder or input_folder
        
        # Create output directory if it doesn't exist
        if self.output_folder and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def convert_file(self, pdf_path, output_path=None):
        """
        Convert a single PDF file to text.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_path (str, optional): Path for the output text file
            
        Returns:
            str: Path to the created text file
        """
        if not output_path:
            # Create output path with same name but .txt extension
            filename = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(self.output_folder, f"{name_without_ext}.txt")
        
        # Extract text from PDF
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            
            # Write text to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            print(f"Successfully converted {pdf_path} to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error converting {pdf_path}: {str(e)}")
            return None
    
    def convert_folder(self):
        """
        Convert all PDF files in the input folder to text files.
        
        Returns:
            list: Paths to all created text files
        """
        if not self.input_folder:
            raise ValueError("Input folder not specified")
        
        created_files = []
        
        # Process all PDF files in the folder
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.input_folder, filename)
                output_file = self.convert_file(pdf_path)
                if output_file:
                    created_files.append(output_file)
        
        return created_files
