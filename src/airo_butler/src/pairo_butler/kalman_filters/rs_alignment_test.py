import cv2
import numpy as np
import pyrealsense2 as rs

from pairo_butler.utils.tools import pyout


def main():
    resolution_color = (640, 480)
    resolution_depth = (640, 480)
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable both color and depth streams
    config.enable_stream(rs.stream.depth, *resolution_depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, *resolution_color, rs.format.rgb8, 30)

    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Align the depth frame to color frame
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color_image_map = np.copy(color_image)

            mask = (depth_image > 200) & (depth_image < 1000)

            color_image[~mask] = np.zeros(3)

            cv2.imshow("RS2", color_image)
            cv2.waitKey(10)

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()
