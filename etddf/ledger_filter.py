from __future__ import division
"""@package etddf

Contains class that records filter inputs and makes available an event triggered buffer.

"""
__author__ = "Luke Barbier"
__copyright__ = "Copyright 2020, COHRINT Lab"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"
__license__ = "MIT"
__maintainer__ = "Luke Barbier"

from copy import deepcopy
from etddf.etfilter import ETFilter, ETFilter_Main
from etddf.asset import Asset
from etddf.ros2python import get_internal_meas_from_ros_meas
import numpy as np
from pdb import set_trace as st
# from etddf.msg import Measurement

class Measurement:
    def __init__(self, meas_type, stamp, src_asset, measured_asset, data, variance, global_pose):

        self.meas_type = meas_type
        self.stamp = stamp
        self.src_asset = src_asset
        self.measured_asset = measured_asset
        self.data = data
        self.variance = variance
        self.global_pose = global_pose

## Lists all measurement substrings to not send implicitly
MEASUREMENT_TYPES_NOT_SHARED =     ["modem", "depth"]
## Indicates meas should be fused on upcoming update step
FUSE_MEAS_NEXT_UPDATE =             -1
## Indicates to use this filters delta multiplier (Won't be used when fusing a buffer from another asset)
THIS_FILTERS_DELTA =                -1

class LedgerFilter:
    """Records filter inputs and makes available an event triggered buffer. """    

    def __init__(self, num_ownship_states, x0, P0, buffer_capacity, meas_space_table, missed_meas_tolerance_table, delta_codebook_table, delta_multiplier, is_main_filter, my_id):
        """Constructor

        Arguments:
            num_ownship_states {int} -- Number of ownship states for each asset
            x0 {np.ndarray} -- initial states
            P0 {np.ndarray} -- initial uncertainty
            buffer_capacity {int} -- capacity of measurement buffer
            meas_space_table {dict} -- Hash that stores how much buffer space a measurement takes up. Str (meas type) -> int (buffer space)
                Must have key entries "bookend", "bookstart" to indicate space needed for measurement implicitness filling in
            missed_meas_tolerance_table {dict} -- Hash that determines how many measurements of each type do we need to miss before indicating a bookend
            delta_codebook_table {dict} -- Hash thatp stores delta trigger for each measurement type. Str(meas type) -> float (delta trigger)
            delta_multiplier {float} -- Delta trigger constant multiplier for this filter
            is_main_filter {bool} -- Is this filter a common or main filter (if main the meas buffer does not matter)
            my_id {int} -- ID# of the current asset (typically 0)
        """
        if delta_multiplier <= 0:
            raise ValueError("Delta Multiplier must be greater than 0")

        self.original_estimate = [deepcopy(x0), deepcopy(P0)]
        self.delta_codebook_table = delta_codebook_table
        self.delta_multiplier = delta_multiplier
        self.buffer = MeasurementBuffer(meas_space_table, buffer_capacity)
        self.missed_meas_tolerance_table = missed_meas_tolerance_table
        self.is_main_filter = is_main_filter
        self.filter = ETFilter(my_id, num_ownship_states, 3, x0, P0, True)
        self.my_id = my_id

        # Initialize Ledgers
        self.ledger_meas = [] # In internal measurement form
        self.ledger_update_times = [] ## Update times of when correction step executed
        self.expected_measurements = {} # When we don't receive an expected measurement we need to insert a "bookend" into the measurement buffer

    def add_meas(self, ros_meas, src_id, measured_id):
        """Adds and records a measurement to the filter

        Arguments:
            ros_meas {etddf.Measurement.msg} -- The measurement in ROS form
            src_id {int} -- asset ID that took the measurement
            measured_id {int} -- asset ID that was measured (can be any value for ownship measurement)
        """
        ros_meas = deepcopy(ros_meas)
        orig_ros_meas = deepcopy(ros_meas)
        # Get the delta trigger for this measurement
        et_delta = self._get_meas_et_delta(ros_meas)

        # Convert ros_meas to an implicit or explicit internal measurement
        meas = get_internal_meas_from_ros_meas(ros_meas, src_id, measured_id, et_delta)
        orig_meas = deepcopy(meas)

        # Common filter with delta tiering
        if not self.is_main_filter and src_id == self.my_id:

            # Check if this measurement is allowed to be sent implicitly
            l = [x for x in MEASUREMENT_TYPES_NOT_SHARED if x in ros_meas.meas_type]
            if not l:
                marker_name = meas.__class__.__name__ +"_" + str(meas.measured_id)
                # Check for Implicit Update
                if self.filter.check_implicit(meas):

                    # Check if this is the first of the measurement stream, if so, insert a bookstart
                    bookstart = marker_name not in list(self.expected_measurements.keys())
                    
                    if bookstart:
                        self.buffer.insert_marker(ros_meas, ros_meas.stamp, bookstart=True)

                    meas = Asset.get_implicit_msg_equivalent(meas)
                    ros_meas.meas_type += "_implicit"
                    
                # Fuse explicitly
                else:
                    # TODO Check for overflow
                    self.buffer.add_meas(deepcopy(ros_meas))
                
                # Indicate we receieved our expected measurement
                base_meas_type = ros_meas.meas_type.split("_implicit")[0]
                missed_tolerance = deepcopy(self.missed_meas_tolerance_table[base_meas_type])
                self.expected_measurements[marker_name] = [missed_tolerance, orig_ros_meas]

        # Append to the ledger
        self.ledger_meas.append(ros_meas)
        # Fuse measurement
        self.filter.add_meas(meas)

    def predict(self, u, Q, time_delta=1.0, use_control_input=False):
        """Executes filter's prediction step

        Arguments:
            u {np.ndarray} -- control input (num_ownship_states / 2, 1)
            Q {np.ndarray} -- motion/process noise (nstates, nstates)

        Keyword Arguments:
            time_delta {float} -- Amount of time to predict in future (default: {1.0})
            use_control_input {bool} -- Whether to use control input or assume constant velocity (default: {False})
        """
        self.filter.predict(u, Q, time_delta, use_control_input)

    def correct(self, update_time):
        """Execute Correction Step in filter

        Arguments:
            update_time {time} -- Update time to record on the ledger update times
        """
        # print("## Expected ##")
        # for e in self.expected_measurements:
        #     print("{}: {}".format(e, self.expected_measurements[e][1].meas_type))
        
        # Check if we received all of the measurements we were expecting
        for emeas in list(self.expected_measurements.keys()):
            [rx, ros_meas] = self.expected_measurements[emeas]

            # We have reached our tolerance on the number of updates without receiving this measurement
            if rx < 1:
                self.buffer.insert_marker(ros_meas, update_time, bookstart=False)
                del self.expected_measurements[emeas]
            else:
                self.expected_measurements[emeas] = [rx - 1, ros_meas]

        # Run correction step on filter
        self.filter.correct()
        self.ledger_update_times.append(update_time)

    def convert(self, delta_multiplier):
        """Converts the filter to have a new delta multiplier
            
        Arguments:
            delta_multiplier {float} -- the delta multiplier of the new filter
        """
        self.delta_multiplier = delta_multiplier

    def check_overflown(self):
        """Checks whether the filter's buffer has overflown

        Returns:
            bool -- True if buffer has overflown
        """
        return self.buffer.check_overflown()

    def peek(self):
        """ Allows peeking of the buffer

        Returns:
            list -- the current state of the buffer
        """
        return self.buffer.peek()

    def flush_buffer(self, final_time):
        """Returns the event triggered buffer

        Arguments:
            final_time {time} -- the last time measurements were considered to be added to the buffer 

        Returns:
            list -- the flushed buffer of measurements
        """
        self.expected_measurements = {}
        return self.buffer.flush(final_time)

    def reset(self, buffer, ledger_update_times, ledger_meas):
        """Resets a ledger filter with the inputted ledgers

        Arguments:
            buffer {MeasurementBuffer} -- Measurement buffer to be preserved
            ledger_update_times {list} -- Update times
            ledger_meas {list} -- List of measurements at each update time

        Raises:
            ValueError: lengths of ledgers do not match
        """
        self.buffer = deepcopy(buffer)
        self.ledger_update_times = deepcopy(ledger_update_times)

        # Measurement Ledger
        self.ledger_meas = deepcopy(ledger_meas)
        if len(self.ledger_meas)-1 != len(self.ledger_update_times):
            raise ValueError("Meas Ledger does not match length of update times!")
        
    def _get_meas_et_delta(self, ros_meas):
        """Gets the delta trigger for the measurement

        Arguments:
            ros_meas {etddf.Measurement.msg} -- The measurement in ROS form

        Raises:
            KeyError: ros_meas.meas_type not found in the delta_codebook_table

        Returns:
            float -- the delta trigger scaled by the filter's delta multiplier
        """
        # Match root measurement type e.g. "modem_range" with "modem_range_implicit"
        for meas_type in self.delta_codebook_table.keys():
            if meas_type in ros_meas.meas_type:
                return self.delta_codebook_table[meas_type] * self.delta_multiplier
        raise KeyError("Measurement Type " + ros_meas.meas_type + " not found in self.delta_codebook_table")

class MeasurementBuffer:
    """ Manages a delta tier buffer for windowed commmunication """

    def __init__(self, meas_space_table, capacity):
        """Constructor

        Arguments:
            meas_space_table {dict} -- Hash to get how much buffer space a measurement takes up. Str (meas type) -> int (buffer space)
            capacity {int} -- capacity of the buffer
        """
        self.meas_space_table = meas_space_table

        # Make room for the final time marker
        self.capacity = capacity - meas_space_table["final_time"]

        self.buffer = []
        self.size = 0
        self.overflown = False

    def check_overflown(self):
        """Checks whether the filter's buffer has overflown

        Returns:
            bool -- True if buffer has overflown
        """
        return self.overflown

    def add_meas(self, ros_meas):
        """Adds a measurement to the buffer, checks for overflow

        Arguments:
            ros_meas {etddf.Measurement.msg} -- the measurement to add

        Returns:
            overflown {bool} -- whether the buffer has overflown
        """
        self.buffer.append(ros_meas)
        if "bookstart" in ros_meas.meas_type:
            meas_space = self.meas_space_table["bookstart"]
        elif "bookend" in ros_meas.meas_type:
            meas_space = self.meas_space_table["bookend"]
        else:
            meas_space = self.meas_space_table[ros_meas.meas_type]
        self.size += meas_space
        if self.size > self.capacity:
            self.overflown = True
        return self.overflown

    def peek(self):
        """ Allows peeking of the buffer

        Returns:
            list -- the current state of the buffer
        """
        return deepcopy(self.buffer)

    def flush(self, final_time):
        """Returns and clears the buffer

        Arguments:
            final_time {time} -- the last time measurements were considered to be added to the buffer 
        
        Returns:
            buffer {list} -- the flushed buffer
        """
        # Insert the "final_time" marker at the end of the buffer
        final_time_marker = Measurement("final_time", final_time, "","",0.0,0.0,[])
        self.buffer.append(final_time_marker)

        old_buffer = deepcopy(self.buffer)

        # Clear buffer
        self.buffer = []
        self.overflown = False
        self.size = 0

        return old_buffer

    def insert_marker(self, ros_meas, timestamp, bookstart=True):
        """Inserts a bookstart/bookend to the buffer

        A bookstart indicates the start of a measurement stream. This lets the other asset
        receiving the buffer know to generate implicit measurments until the bookend (or end of buffer). 
        Measurement streams are usually started by an explicit measurement, but can be started by
        bookstarts if the bookstart type takes less space.

        A bookend indicates the halt of a measurement stream. This lets the other asset
        receiving the buffer know to stop generating implicit measurments in between explicit ones

        Arguments:
            ros_meas {etddf.Measurement.msg} -- A ros measurement of the type to be halted
            timestamp {time} -- Timestamp to indicate halt of implicit measurements

        Keyword Arguments:
            bookstart {bool} -- whether to generate a bookstart or bookend (default: {True})

        Returns:
            bool -- buffer overflow indicator
        """
        marker = deepcopy(ros_meas)
        # print("Marker: {}".format(marker.meas_type))
        if bookstart:
            marker.meas_type += "_bookstart"
        else:
            marker.meas_type += "_bookend"
        marker.stamp = timestamp
        return self.add_meas(marker)