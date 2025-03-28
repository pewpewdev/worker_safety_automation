import cv2

# Global variables to store polygon points
polygon_points_resized = []  # Points in resized frame coordinates
polygon_points_original = []  # Points in original frame coordinates


def draw_polygon(event, x, y, flags, param):
    global polygon_points_resized

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add clicked point to polygon_points_resized list
        polygon_points_resized.append((x, y))

        # Display the clicked point on the resized frame
        cv2.circle(frame_copy_resized, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Video Frame", frame_copy_resized)


def resize_frame(frame, scale_percent=50):
    # Calculate the new dimensions
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)

    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frame


def get_polygon_points(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    # Calculate the scaling factor
    scale_percent = 30  # Change this to match the scale used in resize_frame() function
    scale_factor = 100 / scale_percent
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    # Read the first frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read video frame")
        return

    global frame_copy_resized
    frame_copy_resized = resize_frame(
        frame, scale_percent
    )  # Resize the frame for display

    # Display the resized first frame
    cv2.imshow("Video Frame", frame_copy_resized)

    # Set mouse callback to handle mouse events (drawing polygon points)
    cv2.setMouseCallback("Video Frame", draw_polygon)

    # Wait for user to draw polygon points on the resized frame
    print("Draw polygon points by clicking on the image. Press 'Enter' when done.")
    cv2.waitKey(0)  # Wait indefinitely until user presses 'Enter'

    # Translate polygon points back to original frame coordinates
    global polygon_points_original
    for point in polygon_points_resized:
        x_original = int(point[0] * scale_factor)
        y_original = int(point[1] * scale_factor)
        polygon_points_original.append((x_original, y_original))

    # Print the polygon points in the specified format
    print("Polygon points (original resolution):", polygon_points_original)

    # Release the video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Example usage:
video_file = "videos/Red_zone_creation.mp4"  # Replace with your video file path

# Call the function to get polygon points from the first frame of the video
get_polygon_points(video_file)
