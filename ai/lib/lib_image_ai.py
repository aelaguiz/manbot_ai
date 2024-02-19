from ai.lib import lib_model
import json

from langchain_core.messages import HumanMessage

import logging

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from io import BytesIO
import base64

import pytesseract
from pytesseract import Output
from PIL import Image
import io

logger = logging.getLogger(__name__)


def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')



    # Enhance the contrast of the image
    # enhancer = ImageEnhance.Contrast(image)
    # image = enhancer.enhance(2)

    image = image.point(lambda x: 0 if x > 150 else 255)

    # image = ImageOps.invert(image)
    
    # Apply a threshold filter to make the image binary
    # image = image.point(lambda x: 0 if x < 100 else 255)

    image = image.filter(ImageFilter.SMOOTH)

    return image

def ocr_conversation(image):
    processed_image = preprocess_image(image)

    # processed_image.show()

    # extracted_text = pytesseract.image_to_string(processed_image)
    data = pytesseract.image_to_data(processed_image, output_type=Output.DICT)
    print(data)

    # Create an empty list to hold chat messages
    chat_messages = []

    # Create an empty list to collect bounding boxes and text
    elements = []

    # Collect all text elements and their bounding box information
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:  # Ignore empty strings
            elements.append({
                'text': text,
                'x': data['left'][i],
                'y': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })

    # Sort elements by their vertical position first
    elements.sort(key=lambda e: e['y'])

    # Initialize variables to track rows of text
    rows = []

    # Group elements into rows based on vertical position
    current_row_y = None
    current_row = []
    for element in elements:
        if current_row_y is None or element['y'] > current_row_y + 25:
            # If it's a new row, sort the current row by X, append to rows, and reset the current row
            if current_row:
                current_row.sort(key=lambda e: e['x'])
                rows.append(current_row)
            current_row = [element]
            current_row_y = element['y']
        else:
            # If it's the same row, add the element to the current row
            current_row.append(element)

    # Don't forget to add the last row after the loop ends
    if current_row:
        current_row.sort(key=lambda e: e['x'])
        rows.append(current_row)

    # Process each row to create chat messages with alignment
    for row in rows:
        alignment_threshold = image.width * 0.2  # Adjust based on observation
        alignment = 'left' if row[0]['x'] < alignment_threshold else 'right'

        row_text = ' '.join([element['text'] for element in row])

        print(f"Row {row_text}: {row[0]['x']}, threshold: {alignment_threshold} = {alignment}")

        # The alignment is determined by the x-coordinate of the first text element in the row
        # alignment = 'left' if row[0]['x'] < (image.width / 2) else 'right'
        # Concatenate the texts within the row
        # Append the concatenated text and alignment to the chat messages list
        chat_messages.append({
            'text': row_text,
            'alignment': alignment
        })

    # Output the chat messages with their alignment

    final_messages = []

    # Initialize variables to keep track of the previous message
    previous_alignment = None
    combined_message = ""

    # Iterate through the chat messages to combine consecutive messages on the same side
    for message in chat_messages:
        # Check if the current message is the same alignment as the previous one
        if message['alignment'] == previous_alignment:
            # Combine the message text with a space if it is the same side
            combined_message += " " + message['text']
        else:
            # If the alignment changed, and the combined message is not empty, save the previous messages
            if combined_message:
                final_messages.append((previous_alignment, combined_message))
            # Start a new combined message with the current one
            combined_message = message['text']
            previous_alignment = message['alignment']

    # Don't forget to append the last combined message
    if combined_message:
        final_messages.append((previous_alignment, combined_message))

    # Output the final combined messages
    for alignment, text in final_messages:
        print(f"{alignment} speaker: {text}")

    return final_messages


def describe_image(image, prompt):
    llm = lib_model.get_image_llm()

    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')


    logger.debug(f"Prompting image with prompt: {prompt}, image of length: {len(image_base64)}")

    msg = llm.invoke([
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                },
                {
                    "type": "text",
                    "text": prompt
                }],
        )
    ])

    
    logger.debug(f"Received response: {msg}")

    return msg.content