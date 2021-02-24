from etddf.measurements import *
import numpy as np

def get_internal_meas_from_ros_meas(ros_meas, src_id, measured_id, et_delta):
    """Converts etddf/Measurement.msg (implicit or explicit) to a class in etddf/measurements.py

    Arguments:
        ros_meas {etddf.Measurement.msg} -- The measurement in ROS form
        src_id {int} -- asset ID that took the measurement
        measured_id {int} -- asset ID that was measured (can be any value for ownship measurement)
        et_delta {float} -- Delta trigger for this measurement

    Raises:
        NotImplementedError: Conversion between measurements forms has not been specified

    Returns:
        etddf.measurements.Explicit -- measurement in filter's internal form
    """
    if "implicit" not in ros_meas.meas_type:
        if ros_meas.meas_type == "depth":
            return GPSz_Explicit(src_id, ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "modem_range" and not ros_meas.global_pose: # check global_pose list empty
            return Range_Explicit(src_id, measured_id, ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "modem_range":
            return RangeFromGlobal_Explicit(measured_id, \
                                            np.array(ros_meas.global_pose).reshape(-1,1),\
                                            ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "modem_azimuth" and not ros_meas.global_pose: # check global_pose list empty
            return Azimuth_Explicit(src_id, measured_id, ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "modem_azimuth":
            return AzimuthFromGlobal_Explicit(measured_id, \
                                            np.array(ros_meas.global_pose).reshape(-1,1),\
                                            ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "dvl_x":
            return Velocityx_Explicit(src_id, ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "dvl_y":
            return Velocityy_Explicit(src_id, ros_meas.data, ros_meas.variance, et_delta)
        # Sonar asset
        elif ros_meas.meas_type == "sonar_x" and not ros_meas.global_pose: 
            return LinRelx_Explicit(src_id, measured_id, ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "sonar_y" and not ros_meas.global_pose:
            return LinRely_Explicit(src_id, measured_id, ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "sonar_z":
            return LinRelz_Explicit(src_id, measured_id, ros_meas.data, ros_meas.variance, et_delta)
        # Sonar Landmark
        elif ros_meas.meas_type == "sonar_x" and ros_meas.global_pose:
            return GPSx_Explicit(src_id, ros_meas.global_pose[0] - ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "sonar_y" and ros_meas.global_pose:
            return GPSy_Explicit(src_id, ros_meas.global_pose[1] - ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "gps_x":
            return GPSx_Explicit(src_id, ros_meas.data, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "gps_y":
            return GPSy_Explicit(src_id, ros_meas.data, ros_meas.variance, et_delta)
        else:
            raise NotImplementedError(str(ros_meas))
    # Implicit Measurement
    else:
        if ros_meas.meas_type == "depth_implicit":
            return GPSz_Implicit(src_id, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "dvl_x_implicit":
            return Velocityx_Implicit(src_id, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "dvl_y_implicit":
            return Velocityy_Implicit(src_id, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "sonar_x_implicit":
            return LinRelx_Implicit(src_id, measured_id, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "sonar_y_implicit":
            return LinRely_Implicit(src_id, measured_id, ros_meas.variance, et_delta)
        elif ros_meas.meas_type == "sonar_z_implicit":
            return LinRelz_Implicit(src_id, measured_id, ros_meas.variance, et_delta)
        else:
            raise NotImplementedError(str(ros_meas))