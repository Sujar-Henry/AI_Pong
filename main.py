import cv2
import numpy as np
import pyautogui

def capture_screen():
    # Create a background subtractor
    mog = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=50, detectShadows=False)

    # Initialize previous ball position and velocity
    previous_ball_position = None
    velocity_x, velocity_y = 0, 0
    alpha = 0.5  # Smoothing factor for velocity

    # Define paddle dimensions (assuming the paddle is vertical)
    paddle_width = 10
    paddle_height = 100

    while True:
        # Capture the full region of the screen
        screenshot = pyautogui.screenshot(region=(500, 370, 600, 450))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Define the smaller region of interest (ROI) for detection
        roi_x, roi_y, roi_w, roi_h = 50, 0, 510, 410  # Adjust these values as needed
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # Apply Gaussian Blur to reduce noise in the ROI
        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

        # Remove background in the ROI
        fg_mask = mog.apply(roi_blur)

        # Apply morphological operations to remove noise in the ROI
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Apply Canny Edge Detection in the ROI
        edges = cv2.Canny(fg_mask, 50, 150)

        # Find Contours in the ROI
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_ct = roi.copy()
        cv2.drawContours(roi_ct, contours, -1, (0, 255, 0), 3)

        # Track the ball in the ROI
        ball_position = None
        for contour in contours:
            # Calculate the area of each contour
            area = cv2.contourArea(contour)
            # Filter out small contours that are not the ball
            if area > 100:  # Adjust this threshold based on the size of the ball
                # Get the bounding box coordinates of the contour
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                # Filter based on aspect ratio and size
                if 0.7 < aspect_ratio < 1.3 and 10 < w < 50 and 10 < h < 50:  # Adjust these thresholds based on the size of the ball
                    # Draw a rectangle around the ball
                    cv2.rectangle(roi_ct, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    ball_position = (x + w // 2, y + h // 2)  # Center of the ball

        # Overlay the processed ROI back onto the full frame
        frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi_ct

        # Predict ball trajectory and move paddle
        if ball_position:
            if previous_ball_position is not None:
                # Calculate velocity with smoothing
                velocity_x = alpha * (ball_position[0] - previous_ball_position[0]) + (1 - alpha) * velocity_x
                velocity_y = alpha * (ball_position[1] - previous_ball_position[1]) + (1 - alpha) * velocity_y

                # Predict future position
                predicted_x = ball_position[0] + velocity_x
                predicted_y = ball_position[1] + velocity_y

                # Ensure the predicted position is within bounds
                screen_x = min(max(500 + roi_x + predicted_x, 0), pyautogui.size().width)
                screen_y = min(max(370 + roi_y + predicted_y, 0), pyautogui.size().height)

                # Adjust the paddle position to use its edge to hit the ball
                if velocity_x > 0:  # Ball moving to the right
                    paddle_x = screen_x - paddle_width // 2
                else:  # Ball moving to the left
                    paddle_x = screen_x + paddle_width // 2

                pyautogui.moveTo(paddle_x, screen_y, duration=0.00001)

            # Update previous ball position
            previous_ball_position = ball_position

        # Display the frame
        cv2.imshow('Screen Capture', frame)
        cv2.setWindowProperty('Screen Capture', cv2.WND_PROP_TOPMOST, 1)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

capture_screen()