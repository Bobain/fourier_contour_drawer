import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import List, Dict, Tuple
import cv2

class FourierDrawerAnimator:
    """
    Handles the visualization using a single unified coordinate system.
    Guarantees perfect mechanical alignment between quadrants.
    """
    def __init__(self, 
                 image_path: str, 
                 x_coeffs: List[Dict], 
                 y_coeffs: List[Dict], 
                 original_points: np.ndarray):
        self.image_path = image_path
        self.x_coeffs = x_coeffs
        self.y_coeffs = y_coeffs
        self.original_points = original_points
        self.trace_x = []
        self.trace_y = []
        
        # 1. Determine Scale
        # Points are centered around 0. Find the bounding box.
        max_x = np.max(np.abs(original_points[:, 0]))
        max_y = np.max(np.abs(original_points[:, 1]))
        self.S = max(max_x, max_y) * 1.5 # Half-size of a quadrant
        
        # 2. Define Quadrant Centers
        self.center_tl = (-self.S, self.S)
        self.center_tr = (self.S, self.S)
        self.center_bl = (-self.S, -self.S)
        self.center_br = (self.S, -self.S)
        
        # 3. Setup Figure
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Grid boundaries
        lim = self.S * 2
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        
        # Drawer Divider lines
        self.ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
        self.ax.axvline(0, color='gray', lw=0.5, alpha=0.3)
        
        # 4. Initialize Artists
        self._setup_image()
        
        # Trace line in BR
        self.trace_line, = self.ax.plot([], [], 'black', lw=2, zorder=10)
        
        # Mechanical Projections
        # Vertical: from TR epicycle tip down to BR trace
        self.proj_v, = self.ax.plot([], [], 'blue', alpha=0.5, lw=1.5, zorder=5)
        # Horizontal: from BL epicycle tip right to BR trace
        self.proj_h, = self.ax.plot([], [], 'red', alpha=0.5, lw=1.5, zorder=5)
        
        # Epicycles
        self.x_circles = [self.ax.add_patch(Circle((0, 0), 0, fill=False, color='blue', alpha=0.2, lw=1.0)) for _ in range(len(self.x_coeffs))]
        self.x_radii, = self.ax.plot([], [], 'blue', alpha=0.5, lw=1.2)
        
        self.y_circles = [self.ax.add_patch(Circle((0, 0), 0, fill=False, color='red', alpha=0.2, lw=1.0)) for _ in range(len(self.y_coeffs))]
        self.y_radii, = self.ax.plot([], [], 'red', alpha=0.5, lw=1.2)

        # Labels
        self.ax.text(self.center_tl[0], self.center_tl[1] + self.S*0.8, "Original", ha='center', fontsize=12)
        self.ax.text(self.center_tr[0], self.center_tr[1] + self.S*0.8, "X Component", ha='center', fontsize=12)
        self.ax.text(self.center_bl[0], self.center_bl[1] + self.S*0.8, "Y Component", ha='center', fontsize=12)
        self.ax.text(self.center_br[0], self.center_br[1] + self.S*0.8, "Intersection Trace", ha='center', fontsize=12)

    def _setup_image(self):
        img = cv2.imread(self.image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Scale image to fit in TL quadrant
            # TL quadrant: [-2S, 0] x [0, 2S]
            # We place it centered at self.center_tl
            extent = [self.center_tl[0] - self.S*0.8, self.center_tl[0] + self.S*0.8, 
                      self.center_tl[1] - self.S*0.8, self.center_tl[1] + self.S*0.8]
            self.ax.imshow(img, extent=extent, zorder=1)

    def _calculate_chain(self, coeffs: List[Dict], t: float, center: Tuple[float, float], is_y_component: bool = False) -> Tuple[np.ndarray, Tuple[float, float]]:
        n = len(coeffs)
        points = np.zeros((n + 1, 2))
        curr_val_x, curr_val_y = 0.0, 0.0
        points[0] = [center[0], center[1]]
        
        for i, c in enumerate(coeffs):
            angle = c['phase'] + (2 * np.pi * c['freq'] * t)
            dx = c['amp'] * np.cos(angle)
            dy = c['amp'] * np.sin(angle)
            curr_val_x += dx
            curr_val_y += dy
            points[i+1] = [center[0] + curr_val_x, center[1] + curr_val_y]
            
        if is_y_component:
            # For Y-component, we swap so the REAL oscillation is vertical.
            # Output value iscurr_val_x (the real sum) but we want to plot it on the y-axis.
            swapped = np.zeros_like(points)
            # subplot x = center_x + imag_part
            swapped[:, 0] = center[0] + (points[:, 1] - center[1])
            # subplot y = center_y + real_part (the sum value)
            swapped[:, 1] = center[1] + (points[:, 0] - center[0])
            tip = (swapped[-1, 0], swapped[-1, 1])
            return swapped, tip
            
        return points, (points[-1, 0], points[-1, 1])

    def update(self, frame: int, total_frames: int):
        t = frame / total_frames
        
        # 1. Update X Chain (Top Right)
        x_chain, x_tip = self._calculate_chain(self.x_coeffs, t, self.center_tr)
        for i, circle in enumerate(self.x_circles):
            circle.center = x_chain[i]
            circle.radius = self.x_coeffs[i]['amp']
        self.x_radii.set_data(x_chain[:, 0], x_chain[:, 1])
        
        # 2. Update Y Chain (Bottom Left)
        y_chain, y_tip = self._calculate_chain(self.y_coeffs, t, self.center_bl, is_y_component=True)
        for i, circle in enumerate(self.y_circles):
            circle.center = y_chain[i]
            circle.radius = self.y_coeffs[i]['amp']
        self.y_radii.set_data(y_chain[:, 0], y_chain[:, 1])
        
        # 3. Intersection Point in BR
        # Global x comes from x_tip.x (which aligns with center_tr.x + val)
        # Global y comes from y_tip.y (which aligns with center_bl.y + val)
        drawn_x = x_tip[0]
        drawn_y = y_tip[1]
        
        self.trace_x.append(drawn_x)
        self.trace_y.append(drawn_y)
        self.trace_line.set_data(self.trace_x, self.trace_y)
        
        # 4. Mechanical Projections (Perfect Truth)
        # Vertical from TR tip to Drawing Point
        self.proj_v.set_data([x_tip[0], drawn_x], [x_tip[1], drawn_y])
        # Horizontal from BL tip to Drawing Point
        self.proj_h.set_data([y_tip[0], drawn_x], [y_tip[1], drawn_y])
        
        return ([self.trace_line, self.x_radii, self.y_radii, self.proj_v, self.proj_h] 
                + self.x_circles + self.y_circles)

    def save_animation(self, output_path: str, frames: int = 300):
        ani = animation.FuncAnimation(self.fig, self.update, frames=frames, 
                                    fargs=(frames,), interval=40, blit=True)
        # Tighter layout for the final frame
        plt.tight_layout()
        ani.save(output_path, writer='pillow', fps=25)
        plt.close(self.fig)
