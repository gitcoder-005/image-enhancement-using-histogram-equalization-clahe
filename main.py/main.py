import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk, filedialog
import os

# STEP 1: Select Image

Tk().withdraw()  # Hide root window
file_path = filedialog.askopenfilename(title="Select an Image")

if file_path == "":
    print(" No file selected")
    exit()

print("Selected file:", file_path)

# STEP 2: Load Image
img = cv2.imread(file_path)

if img is None:
    print(" Error loading image!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# STEP 3: Noise Removal

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# STEP 4: Histogram Equalization

hist_eq = cv2.equalizeHist(blur)

# STEP 5: CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(blur)

# STEP 6: Edge Detection

edges_original = cv2.Canny(gray, 100, 200)
edges_hist = cv2.Canny(hist_eq, 100, 200)
edges_clahe = cv2.Canny(clahe_img, 100, 200)

# STEP 7: Display Images

plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)
plt.title('Original')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.title('Histogram Equalization')
plt.imshow(hist_eq, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.title('CLAHE')
plt.imshow(clahe_img, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.title('Edges Original')
plt.imshow(edges_original, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.title('Edges Hist Equalized')
plt.imshow(edges_hist, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.title('Edges CLAHE')
plt.imshow(edges_clahe, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# STEP 8: Histogram Graphs

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.hist(gray.ravel(), bins=256)
plt.title('Original Histogram')

plt.subplot(3, 1, 2)
plt.hist(hist_eq.ravel(), bins=256)
plt.title('Equalized Histogram')

plt.subplot(3, 1, 3)
plt.hist(clahe_img.ravel(), bins=256)
plt.title('CLAHE Histogram')

plt.tight_layout()
plt.show()

# STEP 9: Save Outputs
output_folder = os.path.dirname(file_path)

cv2.imwrite(os.path.join(output_folder, 'output_hist_eq.jpg'), hist_eq)
cv2.imwrite(os.path.join(output_folder, 'output_clahe.jpg'), clahe_img)
cv2.imwrite(os.path.join(output_folder, 'edges_original.jpg'), edges_original)
cv2.imwrite(os.path.join(output_folder, 'edges_hist.jpg'), edges_hist)
cv2.imwrite(os.path.join(output_folder, 'edges_clahe.jpg'), edges_clahe)

print(" Processing complete. Images saved in same folder as input image.")

input("Press Enter to exit...")
