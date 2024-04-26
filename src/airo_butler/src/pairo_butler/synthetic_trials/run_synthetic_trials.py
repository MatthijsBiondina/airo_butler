from pairo_butler.kalman_filters.kalman_filter import KalmanFilter
from pairo_butler.synthetic_trials.data_samples_generator import SyntheticDataGenerator
import rospy as ros

from pairo_butler.utils.tools import load_config


class SyntheticTrials:
    def __init__(self, name: str = "synthetic_trials"):
        self.name = name
        self.config = load_config()

    def start_ros(self):
        ros.init_node(self.name, log_level=ros.INFO)

        ros.loginfo(f"{self.name}: OK!")

    def run(self):
        while not ros.is_shutdown():
            SyntheticDataGenerator.reset()
            KalmanFilter.reset()

            ros.sleep(1 / self.config.rate * self.config.samples_per_trial)


def main():
    node = SyntheticTrials()
    node.start_ros()
    node.run()


if __name__ == "__main__":
    main()
