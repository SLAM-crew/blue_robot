import cv2
import numpy as np

class Camera:
    def __init__(self, type, frame_size):
        if type == 'left':
            self.dist_coeffs = np.array([-0.41728863,  0.22615515, -0.00167113,  0.00549296, -0.03307888])
            self.camera_matrix = np.array([[1.20467414e+03, 0.00000000e+00, 9.07854974e+02], 
                          [0.00000000e+00, 1.20123843e+03, 5.52728845e+02], 
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        else:
            self.camera_matrix = np.array([[1.18411182e+03, 0.00000000e+00, 8.88968918e+02], 
                          [0.00000000e+00, 1.17913758e+03, 5.95308841e+02], 
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            self.dist_coeffs = np.array([-0.39212338, 0.17216101, -0.00425378, 0.00462009, 0.00155077])
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, frame_size, 1, frame_size)

    def undistort_and_crop_frame(self, frame):
        # Применяем undistort для коррекции искажений
        frame_undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
        
        x, y, w, h = self.roi
        frame_undistorted = frame_undistorted[y:y+h, x:x+w]
        
        return frame_undistorted