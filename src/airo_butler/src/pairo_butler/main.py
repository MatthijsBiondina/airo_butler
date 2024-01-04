import rospy as ros

class MainStateMachine:
    def __init__(self, name: str = "unfolding_state_machine"):
        self.node_name: str = name

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)


