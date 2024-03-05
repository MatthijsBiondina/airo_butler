import pyzed.sl as sl
import cv2


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Set the resolution
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Set the depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Set the unit of measurement

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        exit(-1)

    # Capture an image
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Get the image data
        image_data = image.get_data()
        cv2.imshow("ZED Image", image_data)
        cv2.waitKey(0)  # Wait for a key press before closing

    # Close the camera
    zed.close()


if __name__ == "__main__":
    main()
