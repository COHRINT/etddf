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
import scipy
import scipy.optimize
from etddf_minau.msg import Measurement

# Just used for debugging
# class Measurement:
#     def __init__(self, meas_type, stamp, src_asset, measured_asset, data, variance, global_pose):

#         self.meas_type = meas_type
#         self.stamp = stamp
#         self.src_asset = src_asset
#         self.measured_asset = measured_asset
#         self.data = data
#         self.variance = variance
#         self.global_pose = global_pose

## Lists all measurement substrings to not send implicitly
MEASUREMENT_TYPES_SHARED =     ["sonar_y", "sonar_x"]

class LedgerFilter:
    """Records filter inputs and makes available an event triggered buffer. """    

    def __init__(self, num_ownship_states, x0, P0, delta_codebook_table, delta_multiplier, is_main_filter, asset2id, my_name):
        """Constructor

        Arguments:
            num_ownship_states {int} -- Number of ownship states for each asset
            x0 {np.ndarray} -- initial states
            P0 {np.ndarray} -- initial uncertainty
            delta_codebook_table {dict} -- Hash thatp stores delta trigger for each measurement type. Str(meas type) -> float (delta trigger)
            delta_multiplier {float} -- Delta trigger constant multiplier for this filter
            is_main_filter {bool} -- Is this filter a common or main filter (if main the meas buffer does not matter)
            asset2id {dict} -- Hash to get the id number of an asset from the string name
            my_name {str} -- Name to loopkup in asset2id the current asset's ID#
            default_meas_variance {dict} -- Hash to get measurement variance
        """
        if delta_multiplier <= 0:
            raise ValueError("Delta Multiplier must be greater than 0")

        self.delta_codebook_table = delta_codebook_table
        self.delta_multiplier = delta_multiplier
        self.is_main_filter = is_main_filter
        self.filter = ETFilter(my_id, num_ownship_states, 3, x0, P0, True)
        self.original_filter = deepcopy(filter)
        self.asset2id = asset2id
        self.my_name = my_name
        self.default_meas_variance = default_meas_variance

        self.last_ledger_shared = 0

        # Initialize ledger with first update
        self.ledger = {}
        self._add_block()
        
    def _add_block(self):
        next_step = len(self.ledger) + 1
        self.ledger = { 
            next_step : {
                "meas": [], 
                "time" : None, 
                "u" : None,
                "Q" : None, 
                "nav_mean" : None, 
                "nav_cov" : None,
            } 
        }

    def _get_meas_ledger_index(self, meas_time):
        for i in range(1,len(self.ledger)+1):
            ledger_time = self.ledger[i]["time"]
            if ledger_time is None:
                return i
            elif meas_time < ledger_time:
                return i

    def _is_shareable(self, src_asset, meas_type):
        if not self.is_main_filter and src_asset == self.my_name:
            if meas_type in MEASUREMENT_TYPES_SHARED:
                return True
        return False

    def add_meas(self, ros_meas):
        """Adds and records a measurement to the filter

        Arguments:
            ros_meas {etddf.Measurement.msg} -- The measurement in ROS form
        """
        ledger_ind = self._get_meas_ledger_index( ros_meas.stamp )

        # Check for Event-Triggering
        if ledger_ind == len(self.ledger) and self._is_shareable(ros_meas.src_asset, ros_meas.meas_type)):

            src_id = self.asset2id[ros_meas.src_asset]
            measured_id = self.asset2id[ros_meas.measured_asset]
            et_delta = self._get_meas_et_delta(ros_meas.meas_type)
            meas = get_internal_meas_from_ros_meas(ros_meas, src_id, measured_id, et_delta)

            if self.filter.check_implicit(meas):
                ros_meas.meas_type += "_implicit"

                
        # Append to the ledger
        self.ledger[ ledger_ind ]["meas"].append( ros_meas )

    @staticmethod
    def run_covariance_intersection(xa, Pa, xb, Pb):
        """Runs covariance intersection on the two estimates A and B

        Arguments:
            xa {np.ndarray} -- mean of A
            Pa {np.ndarray} -- covariance of A
            xb {np.ndarray} -- mean of B
            Pb {np.ndarray} -- covariance of B
        
        Returns:
            c_bar {np.ndarray} -- intersected estimate
            Pcc {np.ndarray} -- intersected covariance
        """
        Pa_inv = np.linalg.inv(Pa)
        Pb_inv = np.linalg.inv(Pb)

        fxn = lambda omega: np.trace(np.linalg.inv(omega*Pa_inv + (1-omega)*Pb_inv))
        omega_optimal = scipy.optimize.minimize_scalar(fxn, bounds=(0,1), method="bounded").x

        Pcc = np.linalg.inv(omega_optimal*Pa_inv + (1-omega_optimal)*Pb_inv)
        c_bar = Pcc.dot( omega_optimal*Pa_inv.dot(xa) + (1-omega_optimal)*Pb_inv.dot(xb))
        return c_bar.reshape(-1,1), Pcc

    def psci(self, x_prior, P_prior, c_bar, Pcc):
        """ Partial State Update all other states of the filter using the result of CI

        Arguments:
            x_prior {np.ndarray} -- This filter's prior estimate (over common states)
            P_prior {np.ndarray} -- This filter's prior covariance 
            c_bar {np.ndarray} -- intersected estimate
            Pcc {np.ndarray} -- intersected covariance

        Returns:
            None
            Updates self.main_filter.filter.x_hat and P, the delta tier's primary estimate
        """
        x = self.filter.x_hat
        P = self.filter.P

        D_inv = np.linalg.inv(Pcc) - np.linalg.inv(P_prior)
        D_inv_d = np.dot( np.linalg.inv(Pcc), c_bar) - np.dot( np.linalg.inv(P_prior), x_prior)
        
        my_id = self.asset2id[self.my_name]
        begin_ind = my_id*self.num_ownship_states
        end_ind = (my_id+1)*self.num_ownship_states

        info_vector = np.zeros( x.shape )
        info_vector[begin_ind:end_ind] = D_inv_d

        info_matrix = np.zeros( P.shape )
        info_matrix[begin_ind:end_ind, begin_ind:end_ind] = D_inv

        posterior_cov = np.linalg.inv( np.linalg.inv( P ) + info_matrix )
        tmp = np.dot(np.linalg.inv( P ), x) + info_vector
        posterior_state = np.dot( posterior_cov, tmp )

        self.filter.x_hat = posterior_state
        self.filter.P = posterior_cov

    def intersect(self, x, P):
        """Runs covariance intersection with main filter's estimate

        Arguments:
            x {np.ndarray} -- other filter's mean
            P {np.ndarray} -- other filter's covariance

        Returns:
            c_bar {np.ndarray} -- intersected estimate
            Pcc {np.ndarray} -- intersected covariance
        """

        my_id = self.asset2id[self.my_name]

        # Slice out overlapping states in main filter
        begin_ind = my_id*self.num_ownship_states
        end_ind = (my_id+1)*self.num_ownship_states
        x_prior = self.filter.x_hat[begin_ind:end_ind].reshape(-1,1)
        P_prior = self.filter.P[begin_ind:end_ind,begin_ind:end_ind]
        P_prior = P_prior.reshape(self.num_ownship_states, self.num_ownship_states)
        
        c_bar, Pcc = LedgerFilter.run_covariance_intersection(x, P, x_prior, P_prior)

        # Update main filter states
        if Pcc.shape != self.filter.P.shape:
            # self.psci(x_prior, P_prior, c_bar, Pcc)
            self.filter.x_hat[begin_ind:end_ind] = c_bar
            self.filter.P[begin_ind:end_ind,begin_ind:end_ind] = P_prior
        else:
            self.filter.x_hat = c_bar
            self.filter.P = Pcc

        return c_bar, Pcc

    def update(self, update_time, u, Q, nav_mean, nav_cov):
        """Execute Prediction & Correction Step in filter

        Arguments:
            update_time {time} -- Update time to record on the ledger update times
            u {np.ndarray} -- control input (num_ownship_states / 2, 1)
            Q {np.ndarray} -- motion/process noise (nstates, nstates)
            nav filter mean
            nav filter covariance
        """
        # Run prediction step
        if len(self.ledger) > 1:
            time_delta = update_time - self.ledger[len(self.ledger) - 1]["time"]
            self.filter.predict(u, Q, time_delta, use_control_input=False)
        
        
        # Add all measurements
        for ros_meas in self.ledger[len(self.ledger)]["meas"]:
            src_id = self.asset2id[ros_meas.src_asset]
            measured_id = self.asset2id[ros_meas.measured_asset]
            et_delta = self._get_meas_et_delta(ros_meas.meas_type)
            meas = get_internal_meas_from_ros_meas(ros_meas, src_id, measured_id, et_delta)
            self.filter.add_meas(meas)

        # Run correction step on filter
        self.filter.correct()

        # Intersect
        c_bar, Pcc = None, None
        if self.is_main_filter:
            c_bar, Pcc = self.intersect(nav_mean, nav_cov)

        # Save all variables of this update
        self.ledger[len(self.ledger)]["time"] = update_time
        self.ledger[len(self.ledger)]["u"] = u
        self.ledger[len(self.ledger)]["Q"] = Q
        self.ledger[len(self.ledger)]["nav_mean"] = nav_mean
        self.ledger[len(self.ledger)]["nav_cov"] = nav_cov
        self._add_block()

        return c_bar, Pcc

    def convert(self, delta_multiplier):
        """Converts the filter to have a new delta multiplier
            
        Arguments:
            delta_multiplier {float} -- the delta multiplier of the new filter
        """
        self.delta_multiplier = delta_multiplier

    def _get_meas_identifier(self, msg, undo=False):
        """ Returns a unique string for that msg

        sonar_x (implicit/explicit) me to agent0 --> sonar_x_me_agent0

        if undo, msg is a "msg_id"/previous call to this function
        """
        if not undo:
            meas_type = msg.meas_type
            if "implicit" in meas_type:
                meas_type = meas_type.split("_implicit")[0]
            identifier = "{}-{}-{}".format(meas_type, msg.src_asset, msg.measured_asset)
            return identifier
        else: # Go from msg (msg_id) ==> ros msg
            meas_type, src_asset, measured_asset = msg.split("-")
            m = Measurement()
            m.meas_type = meas_type
            m.src_asset = src_asset
            m.measured_asset = measured_asset
            return m

    def _get_shareable_meas_dict(self):
        """
        Returns a measurement dict 
            msg_id ==> list of times the msg appears
                   ==> list of explicit measurements
        """
        meas_dict = {}
        for i in range(self.last_ledger_shared, len(self.ledger)):
            for meas in self.ledger[i]["meas"]:
                if self._is_shareable(meas.src_asset, meas.meas_type):
                    msg_id = self._get_meas_identifier(meas)
                    if msg_id not in meas_dict:
                        meas_dict[msg_id] = {"times" : [],"explicit" : []}
                    meas_dict[msg_id]["times"].append( meas.stamp )
                    if "implicit" not in meas.meas_type:
                        meas_dict[msg_id]["explicit"].append( meas )
        return meas_dict

    def _get_meas_dict_from_buffer(self, buffer):
        """
        Returns a measurement dictionary to assist with recreating the measurement sequence
            msg_id ==> list of burst msgs
                   ==> list of epxlicit measurements
        """
        meas_dict = {}
        create_entry = lambda msg_id : meas_dict[msg_id] = {"bursts" : [], "explicit":[]}
        for meas in buffer:
            if "burst" in meas.meas_type:
                meas_type = meas.meas_type.split("_burst")[0]
                if msg_id not in meas_dict:
                    create_entry(msg_id)
                meas_dict[msg_id]["bursts"].append( meas )
            else:
                msg_id = self._get_meas_identifier(meas)
                if msg_id not in meas_dict:
                    create_entry( msg_id )
                meas_dict[msg_id]["explicit"].append( meas )
        return meas_dict

    def _get_bursts(self, times, threshold=3):
        last_time = None
        bursts = [[]]
        for t in times:
            if last_time is None or (t - last_time).to_sec() < threshold
                bursts[-1].append(t)
            else:
                bursts.append([t])
        return bursts

    def _make_burst_msg(self, msg_id, value, start_time, avg_latency_s):
        meas = self._get_meas_identifier(msg_id, undo=True)
        meas.meas_type += "_burst"
        assert avg_latency_s < 10 # Needed for way we are sending
        meas.data = value + (avg_latency_s / 10.0)
        meas.stamp = start_time
        return meas

    def _expand_burst_msg(self, burst_msg):
        """
        Turn a burst msg into many implicit measurements
        """
        assert "burst" in burst_msg.meas_type
        burst_list = []

        num_msgs = int(burst_msg.data)
        avg_latency = ( burst_msg.data - int(burst_msg.data) )*10
        burst_msg.meas_type = burst_msg.meas_type.split("_burst")
        msg_id = self._get_meas_identifier( burst_msg )
        print("Reconstructed msg_id: {}".format(msg_id))
        for i in range(num_msgs):
            new_msg = self._get_meas_identifier(msg_id, undo=True)
            new_msg.stamp = burst_msg.stamp + rospy.Duration( i * avg_latency )
            new_msg.meas_type += "_implicit"
            new_msg.variance = burst_msg.variance
            new_msg.et_delta = burst_msg.et_delta
            burst_list.append( new_msg )
        return burst_list, avg_latency

    def _add_variances(self, buffer):
        for msg in buffer:
            if "_burst" in msg.meas_type:
                meas_type = msg.meas_type.split("_burst")[0]
                msg.variance = self.default_meas_variance[meas_type]
            else:
                msg.variance = self.default_meas_variance[msg.meas_type]
        return buffer

    def _add_etdeltas(self, buffer, delta_multiplier):
        for msg in buffer:
            if "_burst" in msg.meas_type:
                meas_type = msg.meas_type.split("_burst")[0]
                msg.et_delta = self.delta_codebook_table[meas_type] * delta_multiplier
            else:
                msg.et_delta = self.delta_codebook_table[msg.meas_type] * delta_multiplier
        return buffer


    def pull_buffer(self):
        """Returns the event triggered buffer

        Returns:
            list -- the flushed buffer of measurements
        """
        buffer = []
        meas_dict = self._get_shareable_meas_dict()
        for msg_id in meas_dict:
            times = meas_dict[msg_id]["times"] # Should be sorted
            explicit = meas_dict[msg_id]["explicit"]
            bursts = self._get_bursts(times)
            for b in bursts:
                b_numpy = np.array(b)
                cumdiff = b_numpy[1:] - b_numpy[:-1] # Get the adjacent difference
                latencies = [lat.to_sec() for lat in cumdiff]
                start_time = b[0][0]
                burst_msg = self._make_burst_msg(msg_id, len(b), start_time, np.mean(latencies))
                buffer.append( burst_msg )
            buffer.extend( explicit )
        
        self.last_ledger_shared = len(self.ledger) # Update we've sent all before this
        return buffer

    def receive_buffer(self, buffer, delta_multiplier):
        
        # Add variances & et-deltas
        buffer = self._add_variances(buffer)
        buffer = self._add_etdeltas(buffer, delta_multiplier)

        new_buffer = []
        meas_dict = self._get_meas_dict_from_buffer(buffer)
        for msg_id in meas_dict:
            bursts = meas_dict[msg_id]["bursts"]
            explicit = meas_dict[msg_id]["explicit"]
            for b in bursts:
                implicit_meas, avg_latency = self._expand_burst_msg(b)

                # Match any explicit meas
                matched = []
                for i_implicit in implicit_meas:
                    for j_explicit in explicit:
                        meas_imp = implicit_meas[i_implicit]
                        meas_exp = explicit[j_explicit]
                        if abs( ( meas_imp.stamp - meas_exp.stamp ).to_sec() ) < (avg_latency / 2.0):
                            implicit_meas[i_implicit] = meas_exp
                            matched.append(j_explicit)

                # Removed matched from further consideration
                for m in reversed(matched):
                    explicit.pop(m)

                new_buffer.extend(implicit_meas)    

            # We should have matched every explicit measurement
            assert len(explicit) == 0
            
        # Add every measurement
        for m in new_buffer:
            self.add_meas(m)

    def catch_up(self):
        """
        Rewinds the filter and brings it up to the current momment in time
        """

        # Reset the ledger
        ledger = deepcopy(self.ledger)
        self.ledger = []
        self._add_block()

        # Reset the filter
        self.filter = deepcopy(original_filter)

        for i_step in range(len(ledger)):
            meas_list = ledger[i_step]["meas"]
            update_time = ledger[i_step]["time"]
            u = ledger[i_step]["u"]
            Q = ledger[i_step]["Q"]
            nav_mean = ledger[i_step]["nav_mean"]
            nav_mean = ledger[i_step]["nav_cov"]

            for meas in meas_list:
                self.add_meas(meas)
            self.update(update_time, u, Q, nav_mean, nav_cov)

    def get_asset_estimate(self, asset):
        asset_id = self.asset2id(asset)
        begin_ind = asset_id*self.num_ownship_states
        end_ind = (asset_id+1)*self.num_ownship_states
        asset_mean = self.filter.x_hat[begin_ind:end_ind,0]
        asset_unc = self.filter.P[begin_ind:end_ind,begin_ind:end_ind]
        return deepcopy(asset_mean), deepcopy(asset_unc)
        
    def _get_meas_et_delta(self, meas_type):
        """Gets the delta trigger for the measurement

        Arguments:
            meas_type {str} -- The measurement type

        Raises:
            KeyError: ros_meas.meas_type not found in the delta_codebook_table

        Returns:
            float -- the delta trigger scaled by the filter's delta multiplier
        """
        # Match root measurement type e.g. "modem_range" with "modem_range_implicit"
        for mt in self.delta_codebook_table:
            if mt in meas_type:
                return self.delta_codebook_table[mt] * self.delta_multiplier
        raise KeyError("Measurement Type " + meas_type + " not found in self.delta_codebook_table")