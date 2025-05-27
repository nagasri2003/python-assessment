import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simpson

class MobiusStrip:
    """
    A class to model and analyze a Möbius strip using parametric equations.
    
    Attributes:
        R (float): Radius from center to the strip
        w (float): Width of the strip
        n (int): Number of points in the mesh (resolution)
        u_range (np.array): Array of u parameter values
        v_range (np.array): Array of v parameter values
        points (np.array): 3D coordinates of points on the surface
    """
    
    def __init__(self, R=2.0, w=1.0, n=100):
        """
        Initialize the Möbius strip with given parameters.
        
        Args:
            R (float): Radius from center to the strip (default: 2.0)
            w (float): Width of the strip (default: 1.0)
            n (int): Number of points in the mesh (default: 100)
        """
        self.R = R
        self.w = w
        self.n = n
        
        # Create parameter ranges
        self.u_range = np.linspace(0, 2*np.pi, n)
        self.v_range = np.linspace(-w/2, w/2, n)
        
        # Generate the 3D points
        self.points = self._generate_points()
    
    def _generate_points(self):
        """
        Generate the 3D coordinates using parametric equations.
        
        Returns:
            np.array: Array of shape (n, n, 3) containing (x,y,z) coordinates
        """
        u, v = np.meshgrid(self.u_range, self.v_range)
        
        x = (self.R + v * np.cos(u/2)) * np.cos(u)
        y = (self.R + v * np.cos(u/2)) * np.sin(u)
        z = v * np.sin(u/2)
        
        return np.stack((x, y, z), axis=-1)
    
    def calculate_surface_area(self):
        """
        Calculate the surface area using numerical integration.
        
        Returns:
            float: Approximate surface area of the Möbius strip
        """
        # Get the partial derivatives for the surface normal
        u, v = np.meshgrid(self.u_range, self.v_range)
        
        # Partial derivatives with respect to u
        dx_du = -np.sin(u) * (self.R + v * np.cos(u/2)) - v/2 * np.cos(u) * np.sin(u/2)
        dy_du = np.cos(u) * (self.R + v * np.cos(u/2)) - v/2 * np.sin(u) * np.sin(u/2)
        dz_du = v/2 * np.cos(u/2)
        
        # Partial derivatives with respect to v
        dx_dv = np.cos(u/2) * np.cos(u)
        dy_dv = np.cos(u/2) * np.sin(u)
        dz_dv = np.sin(u/2)
        
        # Cross product of partial derivatives (surface normal)
        cross_x = dy_du * dz_dv - dz_du * dy_dv
        cross_y = dz_du * dx_dv - dx_du * dz_dv
        cross_z = dx_du * dy_dv - dy_du * dx_dv
        
        # Magnitude of the cross product
        magnitude = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        
        # Integrate over u and v
        du = 2*np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)
        
        # Perform double integration using Simpson's rule
        area = simpson(simpson(magnitude, dx=dv), dx=du)
        
        return area
    
    def calculate_edge_length(self):
        """
        Calculate the length of the edge/boundary of the Möbius strip.
        
        Returns:
            float: Approximate edge length
        """
        # The edge is where v = ±w/2 (both give same length)
        v = self.w/2
        u = self.u_range
        
        # Derivatives of edge curve (v fixed at w/2)
        dx_du = -np.sin(u) * (self.R + v * np.cos(u/2)) - v/2 * np.cos(u) * np.sin(u/2)
        dy_du = np.cos(u) * (self.R + v * np.cos(u/2)) - v/2 * np.sin(u) * np.sin(u/2)
        dz_du = v/2 * np.cos(u/2)
        
        # Magnitude of derivative vector
        speed = np.sqrt(dx_du**2 + dy_du**2 + dz_du**2)
        
        # Integrate over u
        du = 2*np.pi / (self.n - 1)
        length = simpson(speed, dx=du)
        
        return length
    
    def plot(self):
        """Visualize the Möbius strip in 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = self.points[..., 0]
        y = self.points[..., 1]
        z = self.points[..., 2]
        
        ax.plot_surface(x, y, z, color='cyan', alpha=0.8)
        
        # Plot the edges
        edge1 = self.points[:, 0, :]  # v = -w/2
        edge2 = self.points[:, -1, :]  # v = w/2
        ax.plot(edge1[:, 0], edge1[:, 1], edge1[:, 2], 'r-', linewidth=2)
        ax.plot(edge2[:, 0], edge2[:, 1], edge2[:, 2], 'b-', linewidth=2)
        
        ax.set_title(f'Möbius Strip (R={self.R}, w={self.w})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a Möbius strip with R=2, width=1, resolution=100
    mobius = MobiusStrip(R=2.0, w=1.0, n=100)
    
    # Calculate and print geometric properties
    surface_area = mobius.calculate_surface_area()
    edge_length = mobius.calculate_edge_length()
    
    print(f"Surface Area: {surface_area:.4f}")
    print(f"Edge Length: {edge_length:.4f}")
    
    # Visualize the strip
    mobius.plot()