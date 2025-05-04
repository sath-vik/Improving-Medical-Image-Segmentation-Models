#!/bin/bash
# Create validation directory if it doesn't exist
mkdir -p "/content/drive/MyDrive/Datasets/Fetus_Head/validation_set"
trainingPath="/content/drive/MyDrive/Datasets/Fetus_Head/training_set"
validationPath="/content/drive/MyDrive/Datasets/Fetus_Head/validation_set"
# Read val.txt and move validation files
while IFS= read -r line; do  # Corrected: Added 'do'
    # Remove whitespace/newline characters
    base_name=$(echo "$line" | tr -d '[:space:]')
    # Define image and annotation filenames
    image="${base_name}.png"
    annotation="${base_name}_Annotation.png"
    # Move image and annotation to validation_set
    if [ -f "$trainingPath/$image" ]; then
        mv "$trainingPath/$image" "$validationPath"
    fi
    if [ -f "$trainingPath/$annotation" ]; then
        mv "$trainingPath/$annotation" "$validationPath"
    fi
done < "val.txt"  # Corrected: Input redirection is fine
echo "Validation files moved successfully."