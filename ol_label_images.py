import os
import io
import json
from io import BytesIO
from PIL import Image, ImageDraw
from google.cloud import vision, vision_v1
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BBOX_DIR = os.path.join(BASE_DIR, "/Users/gabrielcha/desktop/bessy/milk-fashion-data/sample-bounding-boxes-horizontal-slices")
RAW_IMAGE_DIR = os.path.join(BASE_DIR, "/Users/gabrielcha/desktop/bessy/milk-fashion-data/milk-fashion-sample/")  # Add directory for raw images
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "/Users/gabrielcha/desktop/bessy/milk-fashion-data/annotations-sample-horizontal-slices/")  # Add directory for JSON outputs
CREDENTIALS_PATH = os.path.join(BASE_DIR, "../credentials/vision-credentials.json")

# Set Google credentials environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Updated based on the detected items in the dataset
CLOTHING_ITEMS = {
    'bag', 'belt', 'boot', 'bowtie', 'bracelet', 'clothing', 'coat',
    'dress', 'fedora', 'footwear', 'glasses', 'glove', 'handbag', 'hat',
    'high heels', 'jacket', 'jeans', 'luggage & bags', 'miniskirt', 'necklace',
    'outerwear', 'pants', 'sandal', 'scarf', 'shirt', 'shoe', 'shorts',
    'skirt', 'suit', 'sun hat', 'sunglasses', 'tie', 'top', 'watch',
    'blouse', 'tshirt', 't-shirt', 'sweater', 'hoodie', 'blazer',
    'trouser', 'gown', 'jumpsuit', 'romper', 'cap', 'beanie', 
    'sneakers', 'purse', 'backpack', 'tote', 'earring', 'sock', 'stocking'
}

class Image_Labeller():
    def __init__(self, num_images=None):
        # Initialize core components with API key

        self.client = vision.ImageAnnotatorClient()
        
        # Define slices for image splitting
        self.slices = [
            [(0, 0), (0, 0), (1, 0.18), (1, 0.18)]      # Topmost quarter
        ] # horizontal quarters

        self.half_slices = [
            [(0, 0), (0, 0), (1, 0.5), (1, 0.5)],  # Top
            [(0, 0.5), (0, 0.5), (1, 1)]  # Bottom
        ] #half slices
        
        self.horizontal_slices = [
            [(0, 0), (0, 0), (0.5, 1), (0.5, 1)],        # Left half
            [(0.5, 0), (0.5, 0), (1, 1), (1, 1)]         # Right half
        ]

        # Process images
        self.results = {}
        
        # Get list of all valid image files
        image_files = [f for f in os.listdir(RAW_IMAGE_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limit number of images if specified
        if num_images is not None:
            image_files = image_files[:num_images]
            print(f"Processing first {num_images} images in directory...")
        else:
            print(f"Processing all {len(image_files)} images in directory...")
        
        # Process each image
        for image_name in image_files:
            try:
                print(f"Processing {image_name}...")
                self.image = self._load_image(image_name)
                self.results[image_name] = self._process_single_image(image_name)
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                self.results[image_name] = None

    def _load_image(self, image_name):
        """Load image from RAW_IMAGE_DIR and return as bytes"""
        image_path = os.path.join(RAW_IMAGE_DIR, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        with open(image_path, 'rb') as image_file:
            return image_file.read()

    def _process_single_image(self, image_name):
        """Helper method to process a single image"""
        self.all_objects = {}
        self.all_objects["first"] = self.first_stage_obj_det()
        self.all_objects["second"] = self.second_stage_obj_det()
        self.image_with_boxes, vertices = self.draw_bounding_boxes()
        
        # Save the processed image
        if self.image_with_boxes:
            output_path = os.path.join(IMAGE_BBOX_DIR, f"processed_{image_name}")
            self.image_with_boxes.save(output_path)
        
        return vertices

    def first_stage_obj_det(self):
        """
        Detects objects in an image using the Google Vision API and retrieves bounding box information.

        Parameters:
        - image: The image to analyze, provided as a Google Vision `Image` object.
        - client: An instance of the Google Vision API client.

        Returns:
        - objects: A list of localized object annotations detected in the image.
        - result: A dictionary mapping object names to their bounding box vertices.

        """ 
        google_image = vision.Image(content=self.image)
        response = self.client.object_localization(image=google_image)

        if response.error.message:
            raise Exception(f"Google Vision API Error: {response.error.message}")

        objects = response.localized_object_annotations

        # Print only total number of objects
        print(f"\n=== First Stage Detection ===")
        print(f"Total objects found: {len(objects)}")
        
        return objects

    def second_stage_obj_det(self):
        ### RETRIEVING ALL REQUIRED IMAGES ###
        pil_img = Image.open(io.BytesIO(self.image))

        # Get images from all slicing strategies
        quarter_images = self.interval_zoom(self.slices)
        half_images = self.interval_zoom(self.half_slices)
        horizontal_images = self.interval_zoom(self.horizontal_slices)
        
        # Combine all sectioned images
        all_sectioned_images = quarter_images + half_images + horizontal_images

        if not all_sectioned_images:
            raise ValueError("No sectioned images were generated.")

        ### RUN SECOND STAGE OD ###
        image_width, image_height = pil_img.size
        all_objects = []
        
        # Create a new dictionary to store the slice type for each object
        # We'll use a combination of name and score as the key since objects aren't hashable
        self.slice_origins = {}

        # Track which slice we're processing
        total_slices = len(self.slices) + len(self.half_slices) + len(self.horizontal_slices)

        # Iterating through all the sectioned images
        for i, image in enumerate(all_sectioned_images):
            response = self.client.object_localization(image=image)

            if response.error.message:
                raise Exception(f"Google Vision API Error: {response.error.message}")

            objects = response.localized_object_annotations
            
            # Determine which slice configuration we're using
            if i < len(self.slices):
                slice_type = "Quarter"
                slice_index = i
                slice_coords = self.slices[slice_index]
            elif i < len(self.slices) + len(self.half_slices):
                slice_type = "Half"
                slice_index = i - len(self.slices)
                slice_coords = self.half_slices[slice_index]
            else:
                slice_type = "Horizontal"
                slice_index = i - (len(self.slices) + len(self.half_slices))
                slice_coords = self.horizontal_slices[slice_index]

            if not objects:
                print(f"\n=== Second Stage Detection - {slice_type} Section {slice_index + 1} ===")
                print("No objects found in this section")
                continue

            print(f"\n=== Second Stage Detection - {slice_type} Section {slice_index + 1} ===")
            print(f"Total objects found: {len(objects)}")

            # Get section coordinates
            slice_start_x = int(slice_coords[0][0] * image_width)
            slice_start_y = int(slice_coords[0][1] * image_height)
            slice_width = int((slice_coords[2][0] - slice_coords[0][0]) * image_width)
            slice_height = int((slice_coords[2][1] - slice_coords[0][1]) * image_height)

            for object_ in objects:
                adjusted_vertices = []
                for vertex in object_.bounding_poly.normalized_vertices:
                    x = slice_start_x + (vertex.x * slice_width)
                    y = slice_start_y + (vertex.y * slice_height)

                    x = max(0, min(image_width, x))
                    y = max(0, min(image_height, y))

                    adjusted_vertices.append(vision_v1.types.Vertex(x=int(x), y=int(y)))

                object_.bounding_poly.vertices = adjusted_vertices
                all_objects.append(object_)
                
                # Store the slice type using object name and score as a unique identifier
                obj_key = f"{object_.name}_{object_.score}"
                self.slice_origins[obj_key] = slice_type

        if not all_objects:
            raise ValueError("No objects detected in the second stage.")

        return all_objects

    def interval_zoom(self, slice_config):
        """
        Crops the given image into sections based on provided slice configuration.
        
        Parameters:
        - slice_config: List of slice coordinates to use
        
        Returns:
        - sectioned_images: List containing the cropped image objects (Google Vision Image).
        """
        try:
            sectioned_images = []

            # Iterate over the defined sections
            for i, vertices in enumerate(slice_config):
                # Crop each section using the crop_image method
                cropped_image = self.crop_image(vertices)
                
                if cropped_image:
                    # Edge case: Convert the image to RGB mode if it has an alpha channel (RGBA)
                    if cropped_image.mode == 'RGBA':
                        cropped_image = cropped_image.convert('RGB')

                    # Convert the PIL Image to bytes
                    byte_array = BytesIO()
                    cropped_image.save(byte_array, format='JPEG')
                    byte_array = byte_array.getvalue()
                    
                    # Convert image to Google Vision-compatible format
                    image = vision.Image(content=byte_array)
                    sectioned_images.append(image)

            return sectioned_images

        except Exception as e:
            print(f"Error during interval zoom: {e}")
            return None
        
    def crop_image(self, vertices):
        """
        Crops the image based on the given bounding box.
        
        Parameters:
        - binary_img (bytes): image file in bytes 
        - vertices (list): A list of tuples [ (left_x, left_y), (top_x, top_y), (right_x, right_y), (bottom_x, bottom_y)] representing the normalized vertices
        - final (boolean): Boolean value that tells whether it is for Final Stage Crop
        - file (str): The file name for saving the processed image. ONLY needed for Final Stage Crop
        - garment_type (str): The type of garment for saving the cropped garment. ONLY needed for Final Stage Crop
        
        Returns:
        - cropped_image (PIL Image): The cropped image object.

        """
        try:
            pil_img = Image.open(BytesIO(self.image))

            # Convert normalize vertices into pixel values
            width, height = pil_img.size

            # Set bbox equal to a tuple of vertices
            bbox = (vertices[0][0], vertices[0][1], vertices[2][0], vertices[2][1])
            if all(v <= 1 for v in bbox):
                left = int(vertices[0][0] * width)
                top = int(vertices[0][1] * height)
                right = int(vertices[2][0] * width)
                bottom = int(vertices[2][1] * height)
                bbox = (left, top, right, bottom)


            # Crop the image based on the bounding box (left, top, right, bottom)
            cropped_image = pil_img.crop(bbox)

            # Show the cropped image
            return cropped_image

        except Exception as e:
            print(f"Error while cropping image: {e}")
            return None
        
    def calculate_iou(self, box1_coords, box2_coords):
        """Calculate Intersection over Union between two bounding boxes"""
        # Convert coordinates to x1,y1,x2,y2 format
        def get_box_coords(coords):
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        
        box1 = get_box_coords(box1_coords)
        box2 = get_box_coords(box2_coords)
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0

    def filter_overlapping_boxes(self, result):
        """Filter out boxes that overlap significantly"""
        IOU_THRESHOLD = 0.60
        
        filtered_result = {}
        items = list(result.items())
        removed_items = set()  # Track items that should be removed
        
        # Convert RepeatedComposite to list
        first_stage_objects = list(self.all_objects["first"]) if self.all_objects["first"] else []
        second_stage_objects = list(self.all_objects["second"]) if self.all_objects["second"] else []
        all_objects = first_stage_objects + second_stage_objects
        
        print("\n=== Filtering Overlapping Boxes ===")
        for i, (name1, coords1) in enumerate(items):
            if name1 in removed_items:  # Skip if this item was marked for removal
                continue
            
            should_keep = True
            for name2, coords2 in items[i+1:]:
                if name2 in removed_items:  # Skip if this item was marked for removal
                    continue
                
                iou = self.calculate_iou(coords1, coords2)
                print(f"\nComparing boxes:")
                print(f"  {name1} & {name2} - IOU: {iou:.2f}")
                
                if iou > IOU_THRESHOLD:
                    print(f"  High overlap detected (>{IOU_THRESHOLD})")
                    conf1 = next((obj.score for obj in all_objects if obj.name == name1), 0)
                    conf2 = next((obj.score for obj in all_objects if obj.name == name2), 0)
                    if conf1 < conf2:
                        print(f"  Keeping {name2} (confidence: {conf2:.2f})")
                        print(f"  Removing {name1} (confidence: {conf1:.2f})")
                        should_keep = False
                        removed_items.add(name1)
                        break
                    else:
                        print(f"  Keeping {name1} (confidence: {conf1:.2f})")
                        print(f"  Removing {name2} (confidence: {conf2:.2f})")
                        removed_items.add(name2)
            
            if should_keep and name1 not in removed_items:
                filtered_result[name1] = coords1
        
        # Add any remaining items that weren't compared (last items in the list)
        for name, coords in items:
            if name not in removed_items and name not in filtered_result:
                filtered_result[name] = coords
        
        print(f"\nFinal number of boxes after filtering: {len(filtered_result)}")
        print(f"Removed items: {removed_items}")
        return filtered_result

    def draw_bounding_boxes(self):
        try:
            # Step 1: Open the image in binary mode
            pil_img = Image.open(io.BytesIO(self.image))
            image_width, image_height = pil_img.size

            # Store all detected garments in a set to remove duplicates. Store final results in results dict
            set_garments = set()
            result = {}
            
            # First collect all valid boxes without drawing
            if self.all_objects["first"]:
                for obj in self.all_objects["first"]:
                    coordinates = []
                    string_output = self.detect_garment(obj.name).content
                    is_garment = "True" in string_output
                    if is_garment and obj.name not in set_garments:
                        name = obj.name
                        vertices = obj.bounding_poly.vertices
                        norm_vertices = obj.bounding_poly.normalized_vertices

                        if len(norm_vertices) > 2:
                            for norm_vertex in norm_vertices:
                                coordinates.append((norm_vertex.x * image_width, norm_vertex.y * image_height))
                        elif len(vertices) > 2:
                            for vertex in vertices:
                                coordinates.append((vertex.x, vertex.y))
                        else:
                            continue

                        if coordinates and len(coordinates) >= 2:
                            result[name] = coordinates
                            set_garments.add(obj.name)

            # Add second stage boxes
            MIN_CONFIDENCE = 0.45
            if self.all_objects["second"]:
                for obj in self.all_objects["second"]:
                    if obj.score < MIN_CONFIDENCE:
                        continue
                    
                    coordinates = []
                    string_output = self.detect_garment(obj.name).content
                    is_garment = "True" in string_output
                    if is_garment and obj.name not in set_garments:
                        name = obj.name
                        vertices = obj.bounding_poly.vertices

                        if len(vertices) > 2:
                            for vertex in vertices:
                                coordinates.append((vertex.x, vertex.y))
                        else:
                            continue

                        if coordinates and len(coordinates) >= 2:
                            result[name] = coordinates
                            set_garments.add(obj.name)

            # Filter overlapping boxes before drawing anything
            filtered_result = self.filter_overlapping_boxes(result)
            
            # Create a fresh drawing surface
            draw = ImageDraw.Draw(pil_img)
            
            # Draw only the filtered boxes
            for name, coordinates in filtered_result.items():
                first_stage_objects = list(self.all_objects["first"]) if self.all_objects["first"] else []
                second_stage_objects = list(self.all_objects["second"]) if self.all_objects["second"] else []
                all_objects = first_stage_objects + second_stage_objects
                obj = next((obj for obj in all_objects if obj.name == name), None)
                if obj:
                    confidence = obj.score
                    
                    # Determine color based on slice origin
                    if obj in first_stage_objects:
                        color = "blue"  # First stage objects are blue
                    else:
                        # Check if this object was detected in a horizontal slice
                        obj_key = f"{obj.name}_{obj.score}"
                        slice_type = getattr(self, 'slice_origins', {}).get(obj_key, "")
                        if slice_type == "Horizontal":
                            color = "green"  # Horizontal slice detections are green
                        else:
                            color = "red"    # Other second stage detections are red
                    
                    draw.polygon(coordinates, outline=color, width=3)
                    draw.text(coordinates[0], f"{name} ({confidence:.2f})", fill=color)

            return pil_img, filtered_result

        except Exception as e:
            print(f"An error occurred in draw_bounding_boxes: {e}")
            print(f"First stage objects: {type(self.all_objects['first'])}")
            print(f"Second stage objects: {type(self.all_objects['second'])}")
            return None, None

    def detect_garment(self, name):
        """
        Determines whether the given name corresponds to a garment by checking against
        the predefined list of clothing items.

        Parameters:
        - name (str): The name of the object to evaluate.

        Returns:
        - result: An object with a 'content' attribute containing "True" if the object 
                  is a garment, "False" otherwise.
        """
        # Convert to lowercase for case-insensitive matching
        item_lower = name.lower()
        
        # Check if the item is in our CLOTHING_ITEMS set or if it contains a clothing item name
        is_garment = item_lower in CLOTHING_ITEMS or any(clothing_item in item_lower for clothing_item in CLOTHING_ITEMS)
        
        # Return a mock response object with content attribute
        return type('Response', (), {'content': 'True' if is_garment else 'False'})()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process images for garment detection')
    parser.add_argument('--num_images', type=int, help='Number of images to process. If not specified, processes all images.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(RAW_IMAGE_DIR, exist_ok=True)
    os.makedirs(IMAGE_BBOX_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    
    # Initialize and run the labeller
    labeller = Image_Labeller(args.num_images)
    
    # Collect all unique detected classes
    all_detected_classes = set()
    
    # Process each image result and save as JSON
    for image_name, vertices in labeller.results.items():
        if vertices:
            all_detected_classes.update(vertices.keys())
            
            # Create JSON data structure
            json_data = {
                "filename": image_name,
                "objects": []
            }
            
            # Get image dimensions
            image_path = os.path.join(RAW_IMAGE_DIR, image_name)
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error getting image dimensions for {image_name}: {e}")
                continue
            
            # Find the objects for this image
            for object_name, coords in vertices.items():
                # Calculate x_min, y_min, x_max, y_max
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]
                
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)
                
                # Find confidence score
                first_stage_objects = list(labeller.all_objects["first"]) if labeller.all_objects["first"] else []
                second_stage_objects = list(labeller.all_objects["second"]) if labeller.all_objects["second"] else []
                all_objects = first_stage_objects + second_stage_objects
                obj = next((obj for obj in all_objects if obj.name == object_name), None)
                confidence = obj.score if obj else 0.0
                
                # Add to JSON
                json_data["objects"].append({
                    "class": object_name,
                    "confidence": round(confidence, 3),
                    "bounding_box": {
                        "x_min": round(x_min, 2),
                        "y_min": round(y_min, 2),
                        "x_max": round(x_max, 2),
                        "y_max": round(y_max, 2)
                    },
                    "normalized_box": {
                        "x_min": round(x_min / img_width, 4),
                        "y_min": round(y_min / img_height, 4),
                        "x_max": round(x_max / img_width, 4),
                        "y_max": round(y_max / img_height, 4)
                    }
                })
            
            # Write JSON file - use image name without extension as filename
            json_filename = os.path.splitext(image_name)[0] + ".json"
            json_path = os.path.join(JSON_OUTPUT_DIR, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"JSON annotations saved for {image_name}")
    
    # Sort classes alphabetically for better readability
    sorted_classes = sorted(all_detected_classes)
    
    # Write unique classes to README.md
    readme_path = os.path.join(os.path.dirname(BASE_DIR), "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Bessy Open-Label\n\n")
        f.write("## Detected Garment Classes\n\n")
        f.write("The following unique garment classes were detected across all processed images:\n\n")
        for i, class_name in enumerate(sorted_classes, 1):
            f.write(f"{i}. {class_name}\n")
        
        f.write("\n\n## Statistics\n\n")
        f.write(f"- Total images processed: {len(labeller.results)}\n")
        f.write(f"- Total unique garment classes detected: {len(sorted_classes)}\n")
        f.write(f"- Bounding box images saved in: {IMAGE_BBOX_DIR}\n")
        f.write(f"- JSON annotations saved in: {JSON_OUTPUT_DIR}\n")
    
    print("\nProcessing complete!")
    print(f"Total images processed: {len(labeller.results)}")
    print(f"Total unique garment classes detected: {len(sorted_classes)}")
    print(f"Unique classes written to {readme_path}")
    print(f"Bounding box images saved in: {IMAGE_BBOX_DIR}")
    print(f"JSON annotations saved in: {JSON_OUTPUT_DIR}")

if __name__ == "__main__":
    main()