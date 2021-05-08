"""@package etddf

Delta Tier class for windowed event triggered communication

"""
__author__ = "Luke Barbier"
__copyright__ = "Copyright 2020, COHRINT Lab"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"
__license__ = "MIT"
__maintainer__ = "Luke Barbier"

from copy import deepcopy
from etddf.ledger_filter import LedgerFilter, MEASUREMENT_TYPES_NOT_SHARED, THIS_FILTERS_DELTA
from etddf.etfilter import ETFilter, ETFilter_Main
from etddf.ros2python import get_internal_meas_from_ros_meas
import time
from pdb import set_trace as st
import numpy as np


## Constant used to indicate each delta tier should scale the measurement's triggering
## threshold using its own delta factor
USE_DELTA_TIERS = THIS_FILTERS_DELTA

# TODO add threading protection between functions
class DeltaTier:
    """Windowed Communication Event Triggered Communication

    Provides a buffer that can be pulled and received from another
    DeltaTier filter in a planned fashion. 
    """

    def __init__(self, num_ownship_states, x0, P0, buffer_capacity, meas_space_table, missed_meas_tolerance_table, delta_codebook_table, delta_multipliers, asset2id, my_name):
        """Constructor

        Arguments:
            num_ownship_states {int} -- Number of ownship states for each asset
            x0 {np.ndarray} -- initial states
            P0 {np.ndarray} -- initial uncertainty
            buffer_capacity {int} -- capacity of measurement buffer
            meas_space_table {dict} -- Hash that stores how much buffer space a measurement takes up. Str (meas type) -> int (buffer space)
            delta_codebook_table {dict} -- Hash that stores delta trigger for each measurement type. Str(meas type) -> float (delta trigger)
            delta_multipliers {list} -- List of delta trigger multipliers
            asset2id {dict} -- Hash to get the id number of an asset from the string name
            my_name {str} -- Name to loopkup in asset2id the current asset's ID#
            default_meas_variance {dict} -- Hash to get measurement variance
        """
        self.ledger_update_times = []
        self.my_name = my_name

        # Initialize Common Ledger Filters
        self.delta_tiers = {}
        for multiplier in delta_multipliers:
            self.delta_tiers[int(multiplier)] = LedgerFilter(
                num_ownship_states, x0, P0, \
                delta_codebook_table, int(multiplier), \
                False, asset2id, my_name, \
                default_meas_variance
            )

        # Initialize Main Filter
        self.main_filter = LedgerFilter(
            num_ownship_states, x0, P0, \
            delta_codebook_table, 12, \
            True, asset2id, my_name, \
            default_meas_variance
        )

        # Remember for instantiating new LedgerFilters
        self.num_ownship_states = num_ownship_states
        self.buffer_capacity = buffer_capacity
        self.meas_space_table = meas_space_table

    def add_meas(self, ros_meas):
        """Adds a measurement to all ledger filters.
        
        If Measurement is after last correction step, it will be fused on next correction step

        Arguments:
            ros_meas {etddf.Measurement.msg} -- Measurement taken

        Keyword Arguments:
        """
        self.main_filter.add_meas(ros_meas)
        for delta in self.delta_tiers.keys():
            self.delta_tiers[delta].add_meas(ros_meas)

    def receive_buffer(self, shared_buffer, delta_multiplier):
        """Updates main estimate and common estimate using the shared buffer

        Arguments:            
            shared_buffer {list} -- buffer shared from another asset
            delta_multiplier {float} -- multiplier to scale et_delta's with
        """
        self.main_filter.receive_buffer(shared_buffer, delta_multiplier)

        for delta in self.delta_tiers:
            self.delta_tiers[delta].receive_buffer(shared_buffer, delta_multiplier)

    def catch_up(self):
        self.main_filter.catch_up()
        for delta in self.delta_tiers:
            self.delta_tiers[delta].catch_up()

    def pull_buffer(self):
        """Pulls lowest delta multiplier's buffer that hasn't overflown

        Returns:
            multiplier {float} -- the delta multiplier that was chosen
            buffer {list} -- the buffer of ros measurements
        """
        min_delta = None
        buffers = {}

        # Find min valid delta
        for delta in self.delta_tiers:
            buffer = self.delta_tiers[delta].pull_buffer()
            size = self._get_buffer_size(buffer)
            if size > self.buffer_capacity: # Overflown
                continue
            elif min_delta is None or delta < min_delta:
                min_delta = delta
            buffers[delta] = buffer

        # No deltatier is valid, pick the largest and trim
        if min_delta is None:
            print("### ERROR all delta tiers have overflown...handling ###")
            min_delta = max(list(self.delta_tiers.keys()))
            buffer = buffers[min_delta]
            while self._get_buffer_size(buffer) > self.buffer_capacity:
                buffer.pop(-1)

        # Delete delta tiers not chosen
        for delta in self.delta_tiers:
            if delta != min_delta:
                del self.delta_tiers[delta]
                self.delta_tiers[delta] = deepcopy(self.delta_tiers[min_delta])
                self.delta_tiers[delta].convert(delta)

        return buffers[min_delta]

    def update(self, update_time, u, Q, nav_mean, nav_cov):
        """Executes correction step on all filters

        Arguments:
            update_time {time} -- Update time to record on the ledger update times
            u {np.ndarray} -- control input (num_ownship_states / 2, 1)
            Q {np.ndarray} -- motion/process noise (nstates, nstates)
            nav filter mean
            nav filter covariance
        """
        c_bar, Pcc = self.main_filter.update(update_time, u, Q, nav_mean, nav_cov)
        for delta in self.delta_tiers:
            self.delta_tiers[delta].update(update_time, u, Q, nav_mean, nav_cov)
        return c_bar, Pcc

    def get_asset_estimate(self, asset_name):
        """Gets main filter's estimate of an asset

        Arguments:
            asset_name {str} -- Name of asset

        Returns
            np.ndarray -- Mean estimate of asset (num_ownship_states, 1)
            np.ndarray -- Covariance of estimate of asset (num_ownship_states, num_ownship_states)
        """
        return self.main_filter.get_asset_estimate(asset_name)

    def _get_buffer_size(self, buffer):
        count = 0
        for meas in buffer:
            if "burst" in meas.meas_type:
                count += self.meas_space_table["burst"]
            else:
                count += self.meas_space_table[meas.meas_type]
        return count