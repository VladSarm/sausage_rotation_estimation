import os
import re
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from PIL import Image

class RotationOrganizer:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = self.get_sorted_images()
        self.current_idx = 0
        self.start_idx = 0
        self.end_idx = None
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_ui()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def get_sorted_images(self):
        """Get all image files sorted by name."""
        image_files = [f for f in os.listdir(self.image_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Custom sorting to handle image_x_y.jpg format
        def sort_key(filename):
            match = re.match(r'image_(\d+)_(\d+)\..*', filename)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            return filename
            
        return sorted(image_files, key=sort_key)
    
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'right' or event.key == 'd':
            self.next_image(event)
        elif event.key == 'left' or event.key == 'a':
            self.prev_image(event)
        elif event.key == 'm' or event.key == 'space':
            self.mark_360_point(event)
        elif event.key == 'page_down':
            # Jump forward 10 images
            for _ in range(10):
                if self.current_idx < len(self.images) - 1:
                    self.current_idx += 1
            self.show_current_image()
        elif event.key == 'page_up':
            # Jump backward 10 images
            for _ in range(10):
                if self.current_idx > 0:
                    self.current_idx -= 1
            self.show_current_image()
        elif event.key == 'home':
            # Jump to first image
            self.current_idx = 0
            self.show_current_image()
        elif event.key == 'end':
            # Jump to last image
            self.current_idx = len(self.images) - 1
            self.show_current_image()
    
    def setup_ui(self):
        """Setup the UI components."""
        plt.subplots_adjust(bottom=0.2)
        
        # Add navigation buttons
        ax_prev = plt.axes([0.2, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.4, 0.05, 0.1, 0.075])
        ax_mark = plt.axes([0.6, 0.05, 0.2, 0.075])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_mark = Button(ax_mark, 'Mark 360° Point')
        
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_mark.on_clicked(self.mark_360_point)
        
        self.show_current_image()
        
    def show_current_image(self):
        """Display the current image."""
        self.ax.clear()
        img_path = os.path.join(self.image_folder, self.images[self.current_idx])
        img = np.array(Image.open(img_path))
        self.ax.imshow(img)
        self.ax.set_title(f"Image {self.current_idx+1}/{len(self.images)}: {self.images[self.current_idx]}")
        self.ax.axis('off')
        plt.draw()
    
    def next_image(self, event):
        """Go to next image."""
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.show_current_image()
    
    def prev_image(self, event):
        """Go to previous image."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_image()
    
    def mark_360_point(self, event):
        """Mark the current image as completing a 360° rotation."""
        self.end_idx = self.current_idx
        print(f"Marked image {self.images[self.end_idx]} as 360° point")
        
        # Process the images between start_idx and end_idx
        self.process_rotation()
        
        # Set up for next rotation
        self.start_idx = self.end_idx + 1
        if self.start_idx < len(self.images):
            self.current_idx = self.start_idx
            self.show_current_image()
            print(f"Starting new rotation from image {self.images[self.start_idx]}")
        else:
            print("All images processed!")
    
    def process_rotation(self):
        """Distribute images into degree folders."""
        if self.end_idx <= self.start_idx:
            print("No images to process")
            return
            
        # Calculate number of images in this rotation
        num_images = self.end_idx - self.start_idx + 1
        images_to_process = self.images[self.start_idx:self.end_idx + 1]
        
        print(f"Processing {num_images} images for one rotation")
        
        # Create degree folders if they don't exist
        for degree in range(360):
            degree_folder = os.path.join(self.image_folder, "by_degrees", str(degree))
            os.makedirs(degree_folder, exist_ok=True)
        
        # Distribute images across degree folders
        for i, img_name in enumerate(images_to_process):
            # Calculate which degree this image represents
            degree = int((i / num_images) * 360)
            if degree == 360:  # Handle edge case
                degree = 0
                
            source_path = os.path.join(self.image_folder, img_name)
            dest_folder = os.path.join(self.image_folder, "by_degrees", str(degree))
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, img_name)
            
            # Copy the image to the appropriate degree folder
            shutil.copy2(source_path, dest_path)
            print(f"Copied {img_name} to folder {degree}")

def main():
    image_folder = "/Users/vlad.sarm/Documents/sausage_rotation_estimation/data/raw_data"
    organizer = RotationOrganizer(image_folder)
    plt.show()

if __name__ == "__main__":
    main()