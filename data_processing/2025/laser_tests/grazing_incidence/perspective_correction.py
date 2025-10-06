import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import cv2
import h5py
from scipy.optimize import least_squares
from scipy.linalg import svd
import os
from typing import Tuple, Optional, List, Union
import json
from pathlib import Path


class EllipseDrawer:
    """Interactive ellipse drawing tool using matplotlib"""

    def __init__(self, image_path: Union[Path, str], tranformation_path: Union[Path, str],):
        self.image = cv2.imread(str(image_path),  cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert BGR to RGB for matplotlib
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image_rgb)
        self.ax.set_title("Click and drag to draw an ellipse around the circular object")

        # Ellipse drawing state
        self.start_point = None
        self.current_ellipse = None
        self.ellipse_params = None
        self.is_drawing = False

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Add buttons
        self.add_buttons()

        self._path_to_h5 = tranformation_path if isinstance(tranformation_path, Path) else Path(tranformation_path)


    def add_buttons(self):
        """Add control buttons to the interface"""
        # Fit ellipse button
        ax_fit = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.btn_fit = Button(ax_fit, 'Fit Ellipse')
        self.btn_fit.on_clicked(self.fit_ellipse_callback)

        # Save button
        ax_save = plt.axes([0.81, 0.05, 0.1, 0.04])
        self.btn_save = Button(ax_save, 'Save Matrix')
        self.btn_save.on_clicked(self.save_matrix_callback)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.start_point = (event.xdata, event.ydata)
        self.is_drawing = True

    def on_motion(self, event):
        if not self.is_drawing or event.inaxes != self.ax:
            return

        if self.current_ellipse:
            self.current_ellipse.remove()

        # Create ellipse from start point to current point
        x1, y1 = self.start_point
        x2, y2 = event.xdata, event.ydata

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        self.current_ellipse = patches.Ellipse(
            (center_x, center_y), width, height,
            fill=False, edgecolor='red', linewidth=2
        )
        self.ax.add_patch(self.current_ellipse)
        self.fig.canvas.draw()

    def on_release(self, event):
        if not self.is_drawing:
            return
        self.is_drawing = False

        if self.start_point and event.inaxes == self.ax:
            # Store the ellipse parameters
            x1, y1 = self.start_point
            x2, y2 = event.xdata, event.ydata

            self.ellipse_params = {
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'width': abs(x2 - x1),
                'height': abs(y2 - y1),
                'angle': 0  # Initial angle
            }

    def fit_ellipse_callback(self, event):
        """Fit a more precise ellipse using edge detection"""
        if not self.ellipse_params:
            print("Please draw an initial ellipse first!")
            return

        # Get region of interest
        center_x, center_y = self.ellipse_params['center']
        width, height = self.ellipse_params['width'], self.ellipse_params['height']

        # Expand ROI slightly
        roi_width = int(width * 1.2)
        roi_height = int(height * 1.2)

        x1 = max(0, int(center_x - roi_width / 2))
        y1 = max(0, int(center_y - roi_height / 2))
        x2 = min(self.image.shape[1], int(center_x + roi_width / 2))
        y2 = min(self.image.shape[0], int(center_y + roi_height / 2))

        roi = self.image[y1:y2, x1:x2]

        # Convert to grayscale and find edges
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            if len(largest_contour) >= 5:  # Need at least 5 points to fit ellipse
                # Fit ellipse to contour points
                ellipse_cv = cv2.fitEllipse(largest_contour)

                # Convert back to image coordinates
                (center_x_roi, center_y_roi), (width_roi, height_roi), angle_roi = ellipse_cv

                fitted_center_x = center_x_roi + x1
                fitted_center_y = center_y_roi + y1

                # Remove old ellipse and draw fitted one
                if self.current_ellipse:
                    self.current_ellipse.remove()

                self.current_ellipse = patches.Ellipse(
                    (fitted_center_x, fitted_center_y), width_roi, height_roi,
                    angle=angle_roi, fill=False, edgecolor='blue', linewidth=2
                )
                self.ax.add_patch(self.current_ellipse)

                # Update ellipse parameters
                self.ellipse_params = {
                    'center': (fitted_center_x, fitted_center_y),
                    'width': width_roi,
                    'height': height_roi,
                    'angle': angle_roi
                }

                self.fig.canvas.draw()
                print(f"Fitted ellipse: center=({fitted_center_x:.1f}, {fitted_center_y:.1f}), "
                      f"axes=({width_roi:.1f}, {height_roi:.1f}), angle={angle_roi:.1f}°")
            else:
                print("Could not fit ellipse - not enough edge points found")
        else:
            print("No contours found in the selected region")

    def save_matrix_callback(self, event):
        """Calculate and save the perspective correction matrix"""
        if not self.ellipse_params:
            print("Please draw and fit an ellipse first!")
            return

        corrector = PerspectiveCorrector()
        matrix = corrector.calculate_correction_matrix(
            self.ellipse_params, self.image.shape[:2]
        )

        # Save matrix and metadata
        corrector.save_transformation_matrix(
            matrix, self._path_to_h5,
            self.ellipse_params, self.image.shape
        )

        print("Transformation matrix saved to 'transformation_matrix.h5'")

        # Show corrected image preview
        corrected = corrector.apply_correction(self.image, matrix)

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image with Ellipse")
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        ax2.set_title("Perspective Corrected Image")
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    def show(self):
        """Display the interactive interface"""
        plt.show()


class PerspectiveCorrector:
    """Handles perspective correction calculations and transformations"""

    def calculate_correction_matrix(self, ellipse_params: dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Calculate the perspective correction matrix to transform ellipse to circle

        Args:
            ellipse_params: Dictionary with ellipse parameters
            image_shape: (height, width) of the image

        Returns:
            3x3 perspective transformation matrix
        """
        center_x, center_y = ellipse_params['center']
        width = ellipse_params['width']
        height = ellipse_params['height']
        angle = np.radians(ellipse_params['angle'])

        # Calculate semi-major and semi-minor axes
        a = max(width, height) / 2  # semi-major axis
        b = min(width, height) / 2  # semi-minor axis

        print(f"Ellipse parameters:")
        print(f"  Center: ({center_x:.1f}, {center_y:.1f})")
        print(f"  Semi-major axis: {a:.1f}")
        print(f"  Semi-minor axis: {b:.1f}")
        print(f"  Angle: {np.degrees(angle):.1f}°")

        # Calculate the projection angle (tilt angle of the circular object)
        # For a circle viewed at angle θ, the ratio b/a = cos(θ)
        if a > 0:
            cos_theta = b / a
            cos_theta = np.clip(cos_theta, 0, 1)  # Ensure valid range
            projection_angle = np.arccos(cos_theta)

            print(f"  Calculated projection angle: {np.degrees(projection_angle):.1f}°")
        else:
            projection_angle = 0

        # Create correction matrix
        # This is a simplified approach - in reality, perspective correction is more complex
        # and may require knowing camera parameters and 3D geometry

        # Scale factor to make the ellipse circular
        scale_x = 1.0
        scale_y = a / b if b > 0 else 1.0

        # Rotation to align with axes
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        # Combined transformation matrix
        # First translate to origin, then rotate, then scale, then translate back

        # Translation matrices
        T1 = np.array([
            [1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1]
        ], dtype=np.float32)

        T2 = np.array([
            [1, 0, center_x],
            [0, 1, center_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # Rotation matrix
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Scale matrix
        S = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Combine transformations: T2 * S * R * T1
        matrix = T2 @ S @ R @ T1

        return matrix

    def apply_correction(self, image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply perspective correction to an image"""
        height, width = image.shape[:2]
        corrected = cv2.warpPerspective(image, matrix, (width, height))
        return corrected

    def save_transformation_matrix(self, matrix: np.ndarray, filename: Union[Path, str],
                                   ellipse_params: dict, image_shape: Tuple[int, int]):
        """Save transformation matrix and metadata to HDF5 file"""
        with h5py.File(filename, 'w') as f:
            # Save transformation matrix
            f.create_dataset('transformation_matrix', data=matrix)

            # Save metadata
            metadata = f.create_group('metadata')
            metadata.attrs['center_x'] = ellipse_params['center'][0]
            metadata.attrs['center_y'] = ellipse_params['center'][1]
            metadata.attrs['width'] = ellipse_params['width']
            metadata.attrs['height'] = ellipse_params['height']
            metadata.attrs['angle'] = ellipse_params['angle']
            metadata.attrs['image_height'] = image_shape[0]
            metadata.attrs['image_width'] = image_shape[1]

            # Calculate and save derived parameters
            a = max(ellipse_params['width'], ellipse_params['height']) / 2
            b = min(ellipse_params['width'], ellipse_params['height']) / 2
            if a > 0:
                cos_theta = b / a
                projection_angle = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
            else:
                projection_angle = 0

            metadata.attrs['semi_major_axis'] = a
            metadata.attrs['semi_minor_axis'] = b
            metadata.attrs['projection_angle_degrees'] = projection_angle


class ImageTransformer:
    """Class to load transformation matrix and apply to other images"""

    def __init__(self, matrix_file: str):
        self.matrix_file = matrix_file
        self.matrix = None
        self.metadata = None
        self.load_matrix()

    def load_matrix(self):
        """Load transformation matrix and metadata from HDF5 file"""
        if not os.path.exists(self.matrix_file):
            raise FileNotFoundError(f"Matrix file {self.matrix_file} not found")

        with h5py.File(self.matrix_file, 'r') as f:
            self.matrix = f['transformation_matrix'][:]

            # Load metadata
            metadata_group = f['metadata']
            self.metadata = {}
            for key in metadata_group.attrs:
                self.metadata[key] = metadata_group.attrs[key]

    def transform_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Transform an image using the loaded matrix

        Args:
            image_path: Path to input image
            output_path: Optional path to save transformed image

        Returns:
            Transformed image as numpy array
        """
        if self.matrix is None:
            raise ValueError("No transformation matrix loaded")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Apply transformation
        height, width = image.shape[:2]
        transformed = cv2.warpPerspective(self.matrix, (width, height))

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, transformed)
            print(f"Transformed image saved to {output_path}")

        return transformed

    def get_metadata(self) -> dict:
        """Get metadata about the transformation"""
        return self.metadata.copy() if self.metadata else {}

    def print_info(self):
        """Print information about the loaded transformation"""
        if self.metadata:
            print("Transformation Matrix Information:")
            print(f"  Original ellipse center: ({self.metadata['center_x']:.1f}, {self.metadata['center_y']:.1f})")
            print(f"  Ellipse dimensions: {self.metadata['width']:.1f} x {self.metadata['height']:.1f}")
            print(f"  Ellipse angle: {self.metadata['angle']:.1f}°")
            print(f"  Semi-major axis: {self.metadata['semi_major_axis']:.1f}")
            print(f"  Semi-minor axis: {self.metadata['semi_minor_axis']:.1f}")
            print(f"  Calculated projection angle: {self.metadata['projection_angle_degrees']:.1f}°")
            print(f"  Original image size: {self.metadata['image_width']} x {self.metadata['image_height']}")
        else:
            print("No metadata available")


# Example usage functions
def create_ellipse_tool(image_path: Union[Path, str], transformation_path: Union[Path, str]):
    """Create and show the interactive ellipse drawing tool"""
    try:
        drawer = EllipseDrawer(image_path, transformation_path)
        drawer.show()
        return drawer
    except Exception as e:
        print(f"Error creating ellipse tool: {e}")
        return None


def load_and_transform_image(matrix_file: Union[Path, str], image_path: Union[Path, str], output_path: Union[Path, str] = None):
    """Load transformation matrix and apply to an image"""
    try:
        transformer = ImageTransformer(matrix_file)
        transformer.print_info()

        transformed = transformer.transform_image(image_path, output_path)

        # Display results
        original = cv2.imread(image_path)

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
        plt.title("Perspective Corrected")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return transformed

    except Exception as e:
        print(f"Error transforming image: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("Ellipse-to-Circle Perspective Correction Tool")
    print("============================================")
    print()
    print("Usage:")
    print("1. Create ellipse tool:")
    print("   drawer = create_ellipse_tool('your_image.jpg')")
    print("   - Draw ellipse around circular object")
    print("   - Click 'Fit Ellipse' for better precision")
    print("   - Click 'Save Matrix' to save transformation")
    print()
    print("2. Transform other images:")
    print("   load_and_transform_image('transformation_matrix.h5', 'other_image.jpg', 'corrected.jpg')")
    print()
    print("3. Or use the ImageTransformer class directly:")
    print("   transformer = ImageTransformer('transformation_matrix.h5')")
    print("   corrected = transformer.transform_image('image.jpg')")
    path_to_image = Path(r'input_images/GRAZING_INCIDENCE_ILLUMINATED_ROD_20250910.png')
    path_to_corrected_image = Path(r'output_images') / f'{path_to_image.name}_corrected.png'
    path_to_corrected_image.parent.mkdir(parents=True, exist_ok=True)
    path_to_transformation_h5 =  Path(r'transformation_matrices') / "transformation_matrix_1.h5"
    drawer = create_ellipse_tool(path_to_image, path_to_transformation_h5)
    load_and_transform_image(matrix_file=path_to_transformation_h5, image_path=path_to_image, output_path=path_to_corrected_image)