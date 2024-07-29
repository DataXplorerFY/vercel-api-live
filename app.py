from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from flask import send_file
import io
from collections import Counter
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/api/":{"origins":""}})


model = YOLO('best.pt')

def object(img):
    results = model.predict(img)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    object_classes = []

    for index, row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        conf = float(row[4])# Get confidence score from column 4
        d=int(row[5])
#         c=class_list[d]
        obj_class = "orange" #class_list[d]
        
        # Show confidence score with two decimal places
        confidence_text = f"{obj_class}: {conf:.2f}"
        
        
        # Adjust bounding box and text parameters based on the image and your preferences
        bounding_box_color = (0, 0, 255)  # Red color for the first bounding box
        text_color = (255, 255, 255)  # White color for text
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1
        text_thickness = 2
        text_offset_x = 5  # Adjust horizontal offset for text within box
        text_offset_y = 5  # Adjust vertical offset for text within box

        # Draw bounding box with adjusted color
        cv2.rectangle(img, (x1, y1), (x2, y2), bounding_box_color, 2)

        # Calculate text dimensions for background rectangle
        (text_width, text_height) = cv2.getTextSize(confidence_text, text_font, text_size, text_thickness)[0]
        text_background_width = text_width + 2 * text_offset_x  # Add some padding
        text_background_height = text_height + 2 * text_offset_y

        # Draw background rectangle with adjusted color
        cv2.rectangle(img, (x1, y1 - text_background_height), (x1 + text_background_width, y1), bounding_box_color, -1)  # Filled rectangle

        # Display class and confidence score within the box using adjusted parameters
        text_x = x1 + text_offset_x
        text_y = y1 - text_offset_y  # Adjust y-coordinate for text placement within box
        cv2.putText(img, confidence_text, (text_x, text_y), text_font, text_size, text_color, text_thickness)
        object_classes.append(obj_class)
    return object_classes




def count_objects_in_all_images(images):
    total_orange_count = 0
    object_counts = {}  # Dictionary to store counts of all object classes
    for img in images:
        object_classes = object(img.copy())  # Make a copy of the image for drawing 
        orange_count = object_classes.count("orange")
        total_orange_count += orange_count
        
        # Count objects of each class
        for obj, count in Counter(object_classes).items():
            object_counts[obj] = object_counts.get(obj, 0) + count

    print("Total Orange Count in All Images:", total_orange_count)
    print("Object Counts in All Images:", object_counts)
    
    return total_orange_count, object_counts



#-------------------------------------------------------#
# def count_objects_in_all_images(images):
#     total_orange_count = 0
#     for img in images:
#         object_classes = object(img.copy())  # Make a copy of the image for drawing 
#         orange_count = object_classes.count("orange")
#         total_orange_count += orange_count
#         print(f"Object Count in Image: ")
#         for obj, count in Counter(object_classes).items():
#             print(f"{obj}s: {count}")

#     print("Total Orange Count in All Images:", total_orange_count)
#-----------------------------------------------------------------------#
@app.route('/orange-counting', methods=['POST'])
def orangeDetection():
    if(request.method == 'POST'):
        images = []
        for i in range(4):
            image_key = f"image{i+1}"
            if image_key in request.files:
                image = request.files[image_key]
                img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (640, 640))
                images.append(img)
            else:
                print(f"No image found for key {image_key}")

        if images:
            total_orange_count, _ = count_objects_in_all_images(images)  # Extract total orange count
            return jsonify({"total_orange_count": total_orange_count})
        else:
            return jsonify({"error": "No images loaded for prediction"}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))