import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import os

class ComputerVisionAnalyzer:
    """
    Advanced Computer Vision Analysis for Skin Disease Detection
    Includes feature extraction, visualization, and diagnostic insights
    """
    
    def __init__(self, image_path: str):
        """
        Initialize analyzer with image path
        Args:
            image_path: Path to the image file
        """
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.image_hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        self.image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.image_gray.shape
    
    def extract_color_features(self) -> Dict:
        """
        Extract color-based features from the skin region
        Returns:
            Dictionary with color analysis results
        """
        # Calculate mean color values
        b_mean, g_mean, r_mean = cv2.split(self.image_rgb)
        
        # Color statistics
        color_features = {
            'red_mean': float(np.mean(r_mean)),
            'green_mean': float(np.mean(g_mean)),
            'blue_mean': float(np.mean(b_mean)),
            'red_std': float(np.std(r_mean)),
            'green_std': float(np.std(g_mean)),
            'blue_std': float(np.std(b_mean)),
        }
        
        # HSV analysis
        h, s, v = cv2.split(self.image_hsv)
        color_features.update({
            'hue_mean': float(np.mean(h)),
            'saturation_mean': float(np.mean(s)),
            'value_mean': float(np.mean(v)),
            'saturation_std': float(np.std(s)),
        })
        
        return color_features
    
    def extract_texture_features(self) -> Dict:
        """
        Extract texture features using image processing techniques
        Returns:
            Dictionary with texture analysis results
        """
        # Laplacian (edge detection)
        laplacian = cv2.Laplacian(self.image_gray, cv2.CV_64F)
        laplacian_variance = float(np.var(laplacian))
        laplacian_mean = float(np.mean(np.abs(laplacian)))
        
        # Sobel (gradients)
        sobelx = cv2.Sobel(self.image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.image_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Canny edge detection
        edges = cv2.Canny(self.image_gray, 100, 200)
        edge_density = float(np.sum(edges) / (255 * self.height * self.width))
        
        # Texture features
        texture_features = {
            'laplacian_variance': laplacian_variance,
            'laplacian_mean': laplacian_mean,
            'sobel_magnitude_mean': float(np.mean(sobel_magnitude)),
            'sobel_magnitude_std': float(np.std(sobel_magnitude)),
            'edge_density': edge_density,
            'image_contrast': float(np.std(self.image_gray)),
            'image_brightness': float(np.mean(self.image_gray)),
        }
        
        return texture_features
    
    def extract_morphological_features(self) -> Dict:
        """
        Extract morphological features (shape, size, boundaries)
        Returns:
            Dictionary with morphological analysis results
        """
        # Binary threshold
        _, binary = cv2.threshold(self.image_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {'contours_count': 0, 'largest_area': 0}
        
        # Contour analysis
        areas = [cv2.contourArea(c) for c in contours]
        largest_contour = contours[np.argmax(areas)]
        largest_area = max(areas)
        
        # Contour properties
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(largest_area / hull_area) if hull_area > 0 else 0
        
        # Perimeter
        perimeter = float(cv2.arcLength(largest_contour, True))
        circularity = float(4 * np.pi * largest_area / (perimeter**2)) if perimeter > 0 else 0
        
        # Moments for centroid
        M = cv2.moments(largest_contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
        
        morphological_features = {
            'contours_count': len(contours),
            'largest_area': float(largest_area),
            'total_area': float(sum(areas)),
            'solidity': solidity,
            'circularity': circularity,
            'perimeter': perimeter,
            'centroid_x': cx,
            'centroid_y': cy,
            'aspect_ratio': float(self.width / self.height) if self.height > 0 else 0,
        }
        
        return morphological_features
    
    def detect_skin_abnormalities(self) -> Dict:
        """
        Detect potential abnormalities (redness, scaling, inflammation)
        Returns:
            Dictionary with abnormality detection results
        """
        # Redness detection (high R, low G, low B)
        r_channel = self.image_rgb[:, :, 0].astype(float)
        g_channel = self.image_rgb[:, :, 1].astype(float)
        b_channel = self.image_rgb[:, :, 2].astype(float)
        
        # Redness index
        redness_index = (r_channel - g_channel) / (r_channel + g_channel + 1e-5)
        redness_score = float(np.mean(redness_index[redness_index > 0]))
        
        # Inflammation detection (color variance)
        color_variance = np.std([r_channel, g_channel, b_channel])
        
        # Scaling detection (using texture)
        _, binary = cv2.threshold(self.image_gray, 127, 255, cv2.THRESH_BINARY)
        scaling_regions = cv2.countNonZero(binary)
        scaling_percentage = float(scaling_regions / (self.height * self.width) * 100)
        
        abnormality_features = {
            'redness_score': max(0, redness_score),
            'inflammation_index': float(color_variance),
            'scaling_percentage': scaling_percentage,
            'affected_area_percentage': max(0, scaling_percentage),
            'uniformity_score': 1.0 - (color_variance / 255.0) if color_variance > 0 else 0,
        }
        
        return abnormality_features
    
    def extract_histogram_features(self) -> Dict:
        """
        Extract histogram-based features
        Returns:
            Dictionary with histogram analysis results
        """
        # Calculate histograms for each channel
        hist_r = cv2.calcHist([self.image_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([self.image_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([self.image_rgb], [2], None, [256], [0, 256])
        hist_gray = cv2.calcHist([self.image_gray], [0], None, [256], [0, 256])
        
        # Normalize histograms
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
        
        histogram_features = {
            'histogram_r_mean': float(np.mean(hist_r)),
            'histogram_r_std': float(np.std(hist_r)),
            'histogram_g_mean': float(np.mean(hist_g)),
            'histogram_g_std': float(np.std(hist_g)),
            'histogram_b_mean': float(np.mean(hist_b)),
            'histogram_b_std': float(np.std(hist_b)),
            'histogram_gray_mean': float(np.mean(hist_gray)),
            'histogram_gray_std': float(np.std(hist_gray)),
        }
        
        return histogram_features
    
    def analyze_spatial_distribution(self) -> Dict:
        """
        Analyze spatial distribution of abnormalities
        Returns:
            Dictionary with spatial analysis results
        """
        # Divide image into quadrants
        mid_h, mid_w = self.height // 2, self.width // 2
        
        quadrants = [
            self.image_gray[:mid_h, :mid_w],
            self.image_gray[:mid_h, mid_w:],
            self.image_gray[mid_h:, :mid_w],
            self.image_gray[mid_h:, mid_w:],
        ]
        
        quadrant_means = [float(np.mean(q)) for q in quadrants]
        quadrant_stds = [float(np.std(q)) for q in quadrants]
        
        spatial_features = {
            'quadrant_1_mean': quadrant_means[0],
            'quadrant_2_mean': quadrant_means[1],
            'quadrant_3_mean': quadrant_means[2],
            'quadrant_4_mean': quadrant_means[3],
            'spatial_uniformity': 1.0 - (np.std(quadrant_means) / 255.0) if np.std(quadrant_means) > 0 else 1.0,
            'spatial_variance': float(np.var(quadrant_means)),
        }
        
        return spatial_features
    
    def get_comprehensive_analysis(self) -> Dict:
        """
        Get comprehensive computer vision analysis
        Returns:
            Dictionary with all analysis results
        """
        analysis = {
            'color_features': self.extract_color_features(),
            'texture_features': self.extract_texture_features(),
            'morphological_features': self.extract_morphological_features(),
            'abnormality_detection': self.detect_skin_abnormalities(),
            'histogram_features': self.extract_histogram_features(),
            'spatial_analysis': self.analyze_spatial_distribution(),
        }
        
        return analysis
    
    def visualize_analysis(self, output_dir: str = 'cv_analysis') -> Dict:
        """
        Create visualization of computer vision analysis
        Returns:
            Dictionary with paths to generated visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        visualization_paths = {}
        
        # 1. Color channels visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Color Channel Analysis', fontsize=16)
        
        r, g, b = cv2.split(self.image_rgb)
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(r, cmap='Reds')
        axes[0, 1].set_title('Red Channel')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(g, cmap='Greens')
        axes[0, 2].set_title('Green Channel')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(b, cmap='Blues')
        axes[1, 0].set_title('Blue Channel')
        axes[1, 0].axis('off')
        
        h, s, v = cv2.split(self.image_hsv)
        axes[1, 1].imshow(h, cmap='hsv')
        axes[1, 1].set_title('Hue Channel')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(s, cmap='gray')
        axes[1, 2].set_title('Saturation Channel')
        axes[1, 2].axis('off')
        
        path = os.path.join(output_dir, 'color_analysis.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        visualization_paths['color_analysis'] = path
        
        # 2. Edge detection visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Edge and Texture Analysis', fontsize=16)
        
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        edges = cv2.Canny(self.image_gray, 100, 200)
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('Canny Edge Detection')
        axes[0, 1].axis('off')
        
        laplacian = cv2.Laplacian(self.image_gray, cv2.CV_64F)
        axes[1, 0].imshow(laplacian, cmap='gray')
        axes[1, 0].set_title('Laplacian (Texture)')
        axes[1, 0].axis('off')
        
        sobelx = cv2.Sobel(self.image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.image_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        axes[1, 1].imshow(sobel_magnitude, cmap='gray')
        axes[1, 1].set_title('Sobel Gradient')
        axes[1, 1].axis('off')
        
        path = os.path.join(output_dir, 'edge_texture_analysis.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        visualization_paths['edge_texture_analysis'] = path
        
        # 3. Histogram analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Histogram Analysis', fontsize=16)
        
        hist_r = cv2.calcHist([self.image_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([self.image_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([self.image_rgb], [2], None, [256], [0, 256])
        hist_gray = cv2.calcHist([self.image_gray], [0], None, [256], [0, 256])
        
        axes[0, 0].plot(hist_r, color='r', label='Red')
        axes[0, 0].set_title('Red Channel Histogram')
        axes[0, 0].set_xlabel('Pixel Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        axes[0, 1].plot(hist_g, color='g', label='Green')
        axes[0, 1].set_title('Green Channel Histogram')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        axes[1, 0].plot(hist_b, color='b', label='Blue')
        axes[1, 0].set_title('Blue Channel Histogram')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        axes[1, 1].plot(hist_gray, color='gray', label='Grayscale')
        axes[1, 1].set_title('Grayscale Histogram')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        path = os.path.join(output_dir, 'histogram_analysis.png')
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        visualization_paths['histogram_analysis'] = path
        
        return visualization_paths
    
    def get_severity_assessment(self) -> Dict:
        """
        Assess severity based on computer vision features
        Returns:
            Dictionary with severity assessment
        """
        analysis = self.get_comprehensive_analysis()
        
        # Calculate severity score (0-100)
        redness_score = min(100, analysis['abnormality_detection']['redness_score'] * 100)
        inflammation_score = min(100, analysis['abnormality_detection']['inflammation_index'])
        affected_area_score = analysis['abnormality_detection']['affected_area_percentage']
        
        # Weighted severity
        overall_severity = (redness_score * 0.3 + 
                          inflammation_score * 0.3 + 
                          affected_area_score * 0.4)
        
        # Severity classification
        if overall_severity < 25:
            severity_level = "Mild"
        elif overall_severity < 50:
            severity_level = "Moderate"
        elif overall_severity < 75:
            severity_level = "Severe"
        else:
            severity_level = "Very Severe"
        
        return {
            'overall_severity_score': float(overall_severity),
            'severity_level': severity_level,
            'redness_score': float(redness_score),
            'inflammation_score': float(inflammation_score),
            'affected_area_percentage': float(affected_area_score),
        }


# Example usage
if __name__ == "__main__":
    # Test with a sample image
    test_image_path = "test_image.jpg"
    
    if os.path.exists(test_image_path):
        analyzer = ComputerVisionAnalyzer(test_image_path)
        
        # Get comprehensive analysis
        print("=" * 80)
        print("COMPUTER VISION ANALYSIS")
        print("=" * 80)
        
        analysis = analyzer.get_comprehensive_analysis()
        for category, features in analysis.items():
            print(f"\n{category.upper()}")
            print("-" * 80)
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # Get severity assessment
        severity = analyzer.get_severity_assessment()
        print(f"\nSEVERITY ASSESSMENT")
        print("-" * 80)
        for key, value in severity.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        paths = analyzer.visualize_analysis()
        for name, path in paths.items():
            print(f"  ✓ {name}: {path}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Place your test image as 'test_image.jpg' in the project root")