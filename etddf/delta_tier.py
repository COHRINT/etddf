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
from etddf.ledger_filter import LedgerFilter
from etddf.etfilter import ETFilter, ETFilter_Main
from etddf.ros2python import get_internal_meas_from_ros_meas
import time
from pdb import set_trace as st
import numpy as np


# TODO add threading protection between functions
class DeltaTier:
    """Windowed Communication Event Triggered Communication

    Provides a buffer that can be pulled and received from another
    DeltaTier filter in a planned fashion. 
    """

    def __init__(self, num_ownship_states, x0, P0, buffer_capacity, meas_space_table, delta_codebook_table, delta_multipliers, asset2id, my_name, default_meas_variance):
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
        self.last_shared_index = 1

        # Initialize Common Ledger Filters
        self.delta_multipliers = delta_multipliers
        self.common_filter = LedgerFilter(
            num_ownship_states, x0, P0, \
            delta_codebook_table, 1, \
            False, asset2id, my_name, \
            default_meas_variance
        )

        # Initialize Main Filter
        self.main_filter = LedgerFilter(
            num_ownship_states, x0, P0, \
            delta_codebook_table, 12, \
            True, asset2id, my_name, \
            default_meas_variance, \
            self.common_filter.filter
        )

        # Remember for instantiating new LedgerFilters
        self.num_ownship_states = num_ownship_states
        self.buffer_capacity = buffer_capacity
        self.meas_space_table = meas_space_table

    def add_meas(self, ros_meas, common=False):
        """Adds a measurement to all ledger filters.
        
        If Measurement is after last correction step, it will be fused on next correction step

        Arguments:
            ros_meas {etddf.Measurement.msg} -- Measurement taken

        Keyword Arguments:
        """
        ledger_ind = self.main_filter.add_meas(deepcopy(ros_meas))
        if common:
            ledger_ind2 = self.common_filter.add_meas(deepcopy(ros_meas))
            # self.last_shared_index = max([self.last_shared_index, ledger_ind2])
        return ledger_ind

    def receive_buffer(self, shared_buffer, delta_multiplier, src_asset):
        """Updates main estimate and common estimate using the shared buffer

        Arguments:            
            shared_buffer {list} -- buffer shared from another asset
            delta_multiplier {float} -- multiplier to scale et_delta's with
        """
        explicit_cnt = 0
        # Add all measurements to correct update step
        new_buffer = self.common_filter.fillin_buffer(shared_buffer, delta_multiplier)
        for m in new_buffer:
            self.main_filter.add_meas(m)
            self.common_filter.add_meas(m)
            if "implicit" not in m.meas_type:
                explicit_cnt += 1

        # Trim ledgers for updates
        main_ledger = deepcopy(self.main_filter.ledger)
        common_ledger = deepcopy(self.common_filter.ledger)
        self.common_filter.ledger = {}
        self.main_filter.ledger = {}
        for i in range(1, self.last_shared_index):
            self.main_filter.ledger[i] = main_ledger[i]
            self.common_filter.ledger[i] = common_ledger[i]
        self.main_filter._add_block()
        self.common_filter._add_block()

        # Change the filter's estimate to self.last_shared_index
        self.common_filter.reset_estimate()
        self.main_filter.reset_estimate()

        shared_indices = []
        for i_step in range(self.last_shared_index, len(main_ledger)):
            # Main Filter
            meas_list = main_ledger[i_step]["meas"]
            update_time = main_ledger[i_step]["time"]
            u = main_ledger[i_step]["u"]
            Q = main_ledger[i_step]["Q"]
            nav_mean = main_ledger[i_step]["nav_mean"]
            nav_cov = main_ledger[i_step]["nav_cov"]

            for meas in meas_list:
                self.main_filter.add_meas(meas)
            self.main_filter.update(update_time, u, Q, nav_mean, nav_cov)

            # Common filter
            meas_list = common_ledger[i_step]["meas"]
            update_time = common_ledger[i_step]["time"]
            u = common_ledger[i_step]["u"]
            Q = common_ledger[i_step]["Q"]
            nav_mean = common_ledger[i_step]["nav_mean"]
            nav_cov = common_ledger[i_step]["nav_cov"]
            for meas in meas_list:
                self.common_filter.add_meas(meas)
                shared_indices.append(i_step)
                
            self.common_filter.update(update_time, u, Q, nav_mean, nav_cov)

        implicit_cnt = len(new_buffer) - explicit_cnt

        # REPORT
        # report_last_shared_time = self.main_filter.ledger[self.last_shared_index]["time"]
        # report_now_last_shared_time = self.main_filter.ledger[max(shared_indices)]["time"]
        # report_duration = report_now_last_shared_time - report_last_shared_time
        # print("******* BUFFER SHARING REPORT FOR {} w/ Delta {}*******".format(self.my_name, delta_multiplier))
        # print("Last shared time: {}".format(report_last_shared_time.to_sec()))
        # print("Sharing duration: {}".format(report_duration.to_sec()))
        # print("Sharing time now: {}".format(report_now_last_shared_time.to_sec()))
        # print("Implicit cnt: {}".format(implicit_cnt))
        # print("Explicit cnt: {}".format(explicit_cnt))
        
        if shared_indices: # sometimes no measurements shared
            self.last_shared_index = max(shared_indices)

        return implicit_cnt, explicit_cnt

    def catch_up(self, start_ind): # Just used with modem meas
        self.main_filter.catch_up(start_ind)
        self.common_filter.catch_up(start_ind)

    def debug_ledger(self, ledger):
        for i in range(1, len(ledger)):
            print("## {} ##".format(i))
            block = ledger[i]
            for j in range(len(block["meas"])):
                m = block["meas"][j]
                msg_id = self.main_filter._get_meas_identifier(m)
                print("\t#{} {} {} {}".format(j, msg_id, m.data, m.stamp.to_sec()))

    def pull_buffer(self):
        """Pulls lowest delta multiplier's buffer that hasn't overflown

        Returns:
            multiplier {float} -- the delta multiplier that was chosen
            buffer {list} -- the buffer of ros measurements
        """
        last_shared_index = self.last_shared_index
        

        main_ledger = self.main_filter.ledger
        common_ledger = self.common_filter.ledger

        self.last_shared_index = len(main_ledger)

        # print("Main Ledger")
        # self.debug_ledger(main_ledger)

        for delta in self.delta_multipliers:
            deltatier = deepcopy(self.common_filter)
            deltatier.convert(delta)
            deltatier.ledger = {}
            for i in range(1, last_shared_index):
                deltatier.ledger[i] = common_ledger[i]
            deltatier._add_block()

            deltatier.reset_estimate()
            for i_step in range(last_shared_index, len(main_ledger)):
                update_time = common_ledger[i_step]["time"]
                u = common_ledger[i_step]["u"]
                Q = common_ledger[i_step]["Q"]
                nav_mean = common_ledger[i_step]["nav_mean"]
                nav_cov = common_ledger[i_step]["nav_cov"]

                # Use the main ledger for measurements to share
                meas_list = main_ledger[i_step]["meas"]
                for meas in meas_list:
                    # print(meas.meas_type)
                    deltatier.add_meas(meas, output=True)
                deltatier.update(update_time, u, Q, nav_mean, nav_cov)
            buffer = deltatier.pull_buffer(last_shared_index)
            # print("Delta Tier Ledger")
            # self.debug_ledger(deltatier.ledger)
            print("Delta: {} size: {}".format(delta, self._get_buffer_size(buffer)))
            if self._get_buffer_size(buffer) < self.buffer_capacity:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@ SELECTINIG {} @@@@@@@@@@@@@@@@@@@@@@@".format(delta))
                self.common_filter = deltatier
                self.common_filter.convert(1)
                return delta, buffer

        print("### ERRROR all delta tiers have overflown...handling ###")
        while self._get_buffer_size(buffer) > self.buffer_capacity:
            buffer.pop(-1)
        return delta, buffer

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
        self.common_filter.update(update_time, u, Q, nav_mean, nav_cov)
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