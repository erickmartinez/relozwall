import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button
from PIL import Image


class EllipseDrawer:
    def __init__(self, image_path):
        # Load the image
        self.img = Image.open(image_path)
        self.img_array = np.array(self.img)

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)

        # Display the image
        self.ax.imshow(self.img_array, cmap='gray' if len(self.img_array.shape) == 2 else None)
        self.ax.set_title('Click and drag to draw an ellipse. Drag edges to resize.')

        # Initialize ellipse parameters
        self.ellipse = None
        self.center = None
        self.width = 0
        self.height = 0
        self.angle = 0

        # State variables
        self.press = None
        self.dragging_center = False
        self.dragging_edge = None

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # Add button to print ellipse parameters
        ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.button = Button(ax_button, 'Get Parameters')
        self.button.on_clicked(self.print_parameters)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        self.press = (event.xdata, event.ydata)

        if self.ellipse is None:
            # Start drawing a new ellipse
            self.center = self.press
            self.ellipse = Ellipse(self.center, 0, 0, fill=False,
                                   edgecolor='red', linewidth=2)
            self.ax.add_patch(self.ellipse)
        else:
            # Check if clicking near center (for moving)
            cx, cy = self.center
            if np.sqrt((event.xdata - cx) ** 2 + (event.ydata - cy) ** 2) < max(self.width, self.height) * 0.2:
                self.dragging_center = True
            else:
                # Dragging edge for resizing
                self.dragging_edge = 'edge'

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return

        if self.ellipse is None:
            return

        if self.dragging_center:
            # Move the ellipse
            dx = event.xdata - self.press[0]
            dy = event.ydata - self.press[1]
            self.center = (self.center[0] + dx, self.center[1] + dy)
            self.ellipse.center = self.center
            self.press = (event.xdata, event.ydata)
        else:
            # Resize the ellipse
            dx = event.xdata - self.center[0]
            dy = event.ydata - self.center[1]
            self.width = abs(dx) * 2
            self.height = abs(dy) * 2
            self.ellipse.width = self.width
            self.ellipse.height = self.height

        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        self.dragging_center = False
        self.dragging_edge = None

    def print_parameters(self, event):
        if self.ellipse is not None:
            major_radius = max(self.width, self.height) / 2
            minor_radius = min(self.width, self.height) / 2
            print("\n" + "=" * 50)
            print("Ellipse Parameters:")
            print("=" * 50)
            print(f"Center: ({self.center[0]:.2f}, {self.center[1]:.2f})")
            print(f"Major radius: {major_radius:.2f} pixels")
            print(f"Minor radius: {minor_radius:.2f} pixels")
            print(f"Width: {self.width:.2f} pixels")
            print(f"Height: {self.height:.2f} pixels")
            print("=" * 50)

            # Store as attributes for later use
            self.major_radius = major_radius
            self.minor_radius = minor_radius
        else:
            print("No ellipse drawn yet!")

    def get_radii(self):
        """Returns major and minor radii"""
        if self.ellipse is not None:
            major_radius = max(self.width, self.height) / 2
            minor_radius = min(self.width, self.height) / 2
            return major_radius, minor_radius
        return None, None


# Usage
if __name__ == "__main__":
    # Replace with your TIF image path
    image_path = r'./input_images/GRAZING_INCIDENCE_ILLUMINATED_ROD_20250910.png'

    drawer = EllipseDrawer(image_path)
    plt.show()

    # After closing the plot, you can get the radii like this:
    major, minor = drawer.get_radii()
    if major is not None:
        print(f"\nFinal major radius: {major:.2f}")
        print(f"Final minor radius: {minor:.2f}")