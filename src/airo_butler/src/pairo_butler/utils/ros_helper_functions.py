import sys
import rospy as ros
from airo_butler.srv import Reset


def invoke_service(service: str, timeout: float = 5.0):
    try:
        ros.wait_for_service(service, timeout=timeout)
    except ros.exceptions.ROSException as e:
        ros.logerr(f"Cannot invoke {service}. Is it available?")
        raise e
        ros.signal_shutdown(f"Cannot invoke {service}.")
        sys.exit(0)

    service = ros.ServiceProxy(service, Reset)
    resp = service()

    if not resp.success:
        ros.logerr(f"Failed to execute {service}.")
        ros.signal_shutdown(f"Failed to execute {service}")
        sys.exit(0)
    return resp.success
