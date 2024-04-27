from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document

import os
import re
import pyarabic.araby as araby


def flip_arabic_line(line):

    def reverse_number(match):
        # This part splits the number from its surrounding non-word characters
        part = match.group(0)
        # Isolate the number from commas and decimals for reversal
        number = re.sub(r'[^\d]+', '', part)
        reversed_number = number[::-1]
        # Replace the original digits with the reversed digits
        return re.sub(r'\d+', reversed_number, part, 1)

    line = line[::-1]
    
    # New line
    new_line = []

    for word in re.split(r'(\s+)', line):

        if re.match(r'[^\w]*[+-]?[\d,]+(?:\.\d+)?[^\w]*', word):
            
            # Remove commas as it makes matching numbers and reversing them easier
            # This could be removed and the function that reverse numbers could 
            # be modified to account for commas
            number = word.replace(',', '')

            # Use regex to find all numbers and apply the reversing function
            number = re.sub(r'[^\w]*[+-]?[\d,]+(?:\.\d+)?[^\w]*', reverse_number, number)
                
            new_line.append(number)

        elif re.search(r'[A-Za-z]+', word):
            new_line.append(word[::-1])

        else:
            new_line.append(word)
    
    # Join the reversed words to form the sentence
    new_line = ''.join(new_line)
    
    return new_line


def flip_arabic_text(text):
    return '\n'.join(flip_arabic_line(l) for l in text.split('\n'))


def clean_arabic_text(text, **kwargs):
    if kwargs.get('diacritics', True): text = araby.strip_diacritics(text)
    if kwargs.get('harakat', True): text = araby.strip_harakat(text)
    if kwargs.get('lastharaka', True): text = araby.strip_lastharaka(text)
    if kwargs.get('shadda', True): text = araby.strip_shadda(text)
    if kwargs.get('small', True): text = araby.strip_small(text)
    if kwargs.get('tashkeel', True): text = araby.strip_tashkeel(text)
    if kwargs.get('tatweel', True): text = araby.strip_tatweel(text)

    text = re.sub(r'اال', 'الا', text)
    text = re.sub(r'اإل', 'الإ', text)
    text = re.sub(r'األ', 'الأ', text)
    text = re.sub(r'الل', 'لال', text)
    text = re.sub('ألول', 'لأول', text)
    return text


def load_pdf(path, remove_footer=True, page_separator='\n\n\n', **kwargs):
    """
    Loads pdf pages using pdfplumber, preprocesses them and then concats them
    into one text with a specific separator.
    """
    pages = PDFPlumberLoader(path).load()

    merged_pages = []
    for p in pages:
        # Flip arabic text
        page_content = flip_arabic_text(p.page_content)

        # Clean arabic text
        page_content = clean_arabic_text(page_content, **kwargs)

        # Convert all line separtors to become below page separator
        page_content = re.sub(r'\n{3,}', '\n\n', page_content)

        # Remove footer line
        page_content = '\n'.join(l for l in page_content.strip().split('\n')[:-1])
        merged_pages.append(page_content)

    merged_pages = page_separator.join(merged_pages)
    
    # Create a langhchain document
    document = Document(page_content=merged_pages,
                        metadata={"source": os.path.split(path)[-1].strip('.pdf'),
                                  "file_path": path})
    
    return document
