import os

# Set the path to your folder
folder_path = r'D:\OCR_DATA\preprocess\T1_val'
output_file = os.path.join(folder_path, 'labels_val.txt')

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

with open(output_file, 'w', encoding='utf-8') as out_f:
    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)

        if ext.lower() in image_extensions:
            image_path = filename
            txt_file = os.path.join(folder_path, name + '.txt')

            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        out_f.write(f"{image_path} {text}\n")
                    else:
                        print(f"⚠️ Empty text file: {txt_file}")
            else:
                print(f"⚠️ Missing text file for: {image_path}")

print(f"\n✅ labels.txt created at: {output_file}")
