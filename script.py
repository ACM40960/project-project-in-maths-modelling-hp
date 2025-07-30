import os
import xml.etree.ElementTree as ET

# Define your class names (all lowercase for consistency)
classes = ["leopard","amurtiger", "leopardcat", "redfox", "weasel", "wildboar"]  

# Function to convert VOC bbox to YOLO format
def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

# Set your folder paths
input_dir = "Annotations"   # Folder with .xml files
output_dir = "labels"       # Folder to save .txt YOLO annotations

os.makedirs(output_dir, exist_ok=True)

# Process each XML file
for filename in os.listdir(input_dir):
    if not filename.endswith(".xml"):
        continue

    xml_path = os.path.join(input_dir, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        print(f"Skipping {filename}: missing size info")
        continue

    w = int(size.find("width").text)
    h = int(size.find("height").text)

    # Prepare output .txt file
    label_filename = filename.replace(".xml", ".txt")
    label_path = os.path.join(output_dir, label_filename)
    with open(label_path, "w") as out_file:
        objects = list(root.iter("object"))
        if not objects:
            print(f"Warning: No objects in {filename}")
        
        for obj in objects:
            cls = obj.find("name").text.strip().lower()
            if cls not in classes:
                print(f"Skipping unknown class '{cls}' in {filename}")
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")

print("✅ Conversion complete. YOLO annotations saved in:", output_dir)

