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
np.set_printoptions(precision=3)
import scipy
import scipy.optimize
from etddf_minau.msg import Measurement
import rospy

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

    def __init__(self, num_ownship_states, x0, P0, delta_codebook_table, delta_multiplier, is_main_filter, asset2id, my_name, default_meas_variance, common_filter=None):
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
            common_filter {dict} -- asset to common ETFilter
        """
        if delta_multiplier <= 0:
            raise ValueError("Delta Multiplier must be greater than 0")

        self.num_ownship_states = num_ownship_states
        self.delta_codebook_table = delta_codebook_table
        self.delta_multiplier = delta_multiplier
        self.is_main_filter = is_main_filter
        if self.is_main_filter:
            assert common_filter is not None
            self.filter = ETFilter_Main(asset2id[my_name], num_ownship_states, 3, x0, P0, True, { "":common_filter})
        else:
            self.filter = ETFilter(asset2id[my_name], num_ownship_states, 3, x0, P0, True)
        self.original_filter = deepcopy(self.filter)
        self.asset2id = asset2id
        self.my_name = my_name
        self.default_meas_variance = default_meas_variance

        # Initialize ledger with first update
        self.ledger = {}
        self._add_block()

        self.explicit_count = 0
        self.meas_types_received = []

    def change_common_filter(self, common_filter):
        self.filter.common_filters = {"" : common_filter}
        
    def _add_block(self):
        next_step = len(self.ledger) + 1
        self.ledger[next_step] = {
            "meas": [], 
            "time" : None, 
            "u" : None,
            "Q" : None, 
            "nav_mean" : None, 
            "nav_cov" : None,
            "x_hat_prior" : deepcopy(self.filter.x_hat),
            "P_prior" : deepcopy(self.filter.P)
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
            for m in MEASUREMENT_TYPES_SHARED:
                if m in meas_type:
                    return True
        return False

    def add_meas(self, ros_meas, output=False):
        """Adds and records a measurement to the filter

        Arguments:
            ros_meas {etddf.Measurement.msg} -- The measurement in ROS form
        """
        msg_id = self._get_meas_identifier(ros_meas)
        # Main filter fuses all measurements
        if self.is_main_filter:
            pass
        elif ros_meas.src_asset != self.my_name:
            pass
        elif self._is_shareable(ros_meas.src_asset, ros_meas.meas_type): 
            pass
        elif msg_id in self.meas_types_received:
            return
        else: # Don't fuse (e.g. depth, sonar_z)
            return -1
        self.meas_types_received.append(msg_id)

        ledger_ind = self._get_meas_ledger_index( ros_meas.stamp )

        # Check for Event-Triggering
        if self._is_shareable(ros_meas.src_asset, ros_meas.meas_type):
            if "implicit" not in ros_meas.meas_type:
                src_id = self.asset2id[ros_meas.src_asset]
                measured_id = self.asset2id[ros_meas.measured_asset]
                ros_meas.et_delta = self._get_meas_et_delta(ros_meas.meas_type)
                meas = get_internal_meas_from_ros_meas(ros_meas, src_id, measured_id)

                implicit, innovation = self.filter.check_implicit(meas)
                if implicit:

                    ros_meas.meas_type += "_implicit"

                    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ IMPLICIT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    # print(ros_meas)
                    # print(vars(meas))
                    # print(self.filter.x_hat)
                else:
                    self.explicit_count += 1
                    if output:
                        expected = meas.data - innovation
                        meas_id = self._get_meas_identifier(ros_meas)
                        last_update_time = self.ledger[len(self.ledger)-1]["time"]
                        # print("Explicit {} {} : expected: {}, got: {}".format(last_update_time.to_sec(), meas_id, expected, meas.data))
                        # print(self.meas_types_received)
                        # print(self.filter.x_hat.T)
                    # print("Explicit #{} {} : {}".format(self.explicit_count, self.delta_multiplier, ros_meas.meas_type))
                #     print(ros_meas)
                #     print(vars(meas))
                #     print(self.filter.x_hat)


                
        # Append to the ledger
        self.ledger[ ledger_ind ]["meas"].append( ros_meas )
        return ledger_ind

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

        # print("Omega: {}".format(omega_optimal)) # We'd expect a value of 1

        Pcc = np.linalg.inv(omega_optimal*Pa_inv + (1-omega_optimal)*Pb_inv)
        c_bar = Pcc.dot( omega_optimal*Pa_inv.dot(xa) + (1-omega_optimal)*Pb_inv.dot(xb))

        jump = max( [np.linalg.norm(c_bar - xa), np.linalg.norm(c_bar - xb)] )

        if jump > 10: # Think this is due to a floating point error in the inversion
            print("!!!!!!!!!!! BIG JUMP!!!!!!!")
            print(xa)
            print(xb)
            print(c_bar)
            print(omega_optimal)
            print(Pa)
            print(Pb)
            print(Pcc)

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
        # Full state estimates
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
            self.psci(x_prior, P_prior, c_bar, Pcc)
            # self.filter.x_hat[begin_ind:end_ind] = c_bar
            # self.filter.P[begin_ind:end_ind,begin_ind:end_ind] = Pcc
        else:
            self.filter.x_hat = c_bar
            self.filter.P = Pcc

        return c_bar, Pcc

    def reset_estimate(self):
        self.filter.x_hat = self.ledger[len(self.ledger)]["x_hat_prior"]
        self.filter.P = self.ledger[len(self.ledger)]["P_prior"]

    def update(self, update_time, u, Q, nav_mean, nav_cov):
        """Execute Prediction & Correction Step in filter

        Arguments:
            update_time {time} -- Update time to record on the ledger update times
            u {np.ndarray} -- control input (num_ownship_states / 2, 1)
            Q {np.ndarray} -- motion/process noise (nstates, nstates)
            nav filter mean
            nav filter covariance
        """
        # self.ledger[len(self.ledger)]["x_hat_prior"] = 
        # self.ledger[len(self.ledger)]["P_prior"] = deepcopy(self.filter.P)
        self.meas_types_received = []

        # Run prediction step
        if len(self.ledger) > 1:
            time_delta = (update_time - self.ledger[len(self.ledger) - 1]["time"]).to_sec()
            self.filter.predict(u, Q, time_delta, use_control_input=False)
        
        
        # Add all measurements
        # if self.my_name != "surface":
            # print("### {} ###".format(self.my_name))
        for ros_meas in self.ledger[len(self.ledger)]["meas"]:

            src_id = self.asset2id[ros_meas.src_asset]
            measured_id = self.asset2id[ros_meas.measured_asset]
            meas = get_internal_meas_from_ros_meas(ros_meas, src_id, measured_id)
            # if self.my_name != "surface":
                # print(self._get_meas_identifier(ros_meas))
            self.filter.add_meas(meas)

        # Run correction step on filter
        self.filter.correct()

        # Intersect
        c_bar, Pcc = None, None
        if self.is_main_filter and (nav_mean is not None and nav_cov is not None):
            # print("***************************8 Intersecting **********************************")
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
            elif "burst" in meas_type:
                meas_type = meas_type.split("_burst")[0]
            identifier = "{}-{}-{}".format(meas_type, msg.src_asset, msg.measured_asset)
            return identifier
        else: # Go from msg (msg_id) ==> ros msg
            meas_type, src_asset, measured_asset = msg.split("-")
            m = Measurement()
            m.meas_type = meas_type
            m.src_asset = src_asset
            m.measured_asset = measured_asset
            return m

    def _get_shareable_meas_dict(self, last_shared_index):
        """
        Returns a measurement dict 
            msg_id ==> list of times the msg appears
                   ==> list of explicit measurements
        """
        meas_dict = {}
        explicit_count = 0
        for i in range(last_shared_index, len(self.ledger)):
            for meas in self.ledger[i]["meas"]:
                if self._is_shareable(meas.src_asset, meas.meas_type):
                    msg_id = self._get_meas_identifier(meas)
                    if msg_id not in meas_dict:
                        meas_dict[msg_id] = {"times" : [],"explicit" : []}
                    meas_dict[msg_id]["times"].append( meas.stamp )
                    if "implicit" not in meas.meas_type:
                        meas_dict[msg_id]["explicit"].append( meas )
                        explicit_count += 1
        # print("Delta: {} | Explicit Count creating meas dict: {}".format(self.delta_multiplier, explicit_count))
        return meas_dict

    def _get_meas_dict_from_buffer(self, buffer):
        """
        Returns a measurement dictionary to assist with recreating the measurement sequence
            msg_id ==> list of burst msgs
                   ==> list of epxlicit measurements
        """
        meas_dict = {}
        for meas in buffer:
            msg_id = self._get_meas_identifier(meas)
            if msg_id not in meas_dict:
                    meas_dict[msg_id] = {"bursts" : [], "explicit":[]}
            if "burst" in meas.meas_type:
                meas_dict[msg_id]["bursts"].append( meas )
            else:
                meas_dict[msg_id]["explicit"].append( meas )
        return meas_dict

    def _get_bursts(self, times, threshold=3):
        last_time = None
        bursts = [[]]
        for t in times:
            if last_time is None or (t - last_time).to_sec() < threshold:
                bursts[-1].append(t)
            else:
                bursts.append([t])
            last_time = t
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
        # burst_msg.meas_type = burst_msg.meas_type.split("_burst")[0]
        msg_id = self._get_meas_identifier( burst_msg )
        # print("Reconstructed msg_id: {}".format(msg_id))
        # print("Avg latency: {}".format(avg_latency))
        # print("Num msgs: {}".format(num_msgs))
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
            else:
                meas_type = msg.meas_type

            msg.variance = self.default_meas_variance[meas_type] * 2.0
        return buffer

    def _add_etdeltas(self, buffer, delta_multiplier):
        for msg in buffer:
            if "_burst" in msg.meas_type:
                meas_type = msg.meas_type.split("_burst")[0]
                msg.et_delta = self.delta_codebook_table[meas_type] * delta_multiplier
            else:
                msg.et_delta = self.delta_codebook_table[msg.meas_type] * delta_multiplier
        # print(buffer)
        return buffer


    def pull_buffer(self, last_shared_index):
        """Returns the event triggered buffer

        Returns:
            list -- the flushed buffer of measurements
        """
        buffer = []
        explicit_buffer = []

        # report_implicit_count = 0
        # report_last_shared_time = self.ledger[last_shared_index]["time"]
        # report_now_last_shared_time = rospy.get_rostime()
        # report_duration = report_now_last_shared_time - report_last_shared_time

        meas_dict = self._get_shareable_meas_dict(last_shared_index)
        print("PULLING BUFFER: current index {}".format(len(self.ledger)))
        for msg_id in meas_dict:
            times = meas_dict[msg_id]["times"] # Should be sorted
            explicit = meas_dict[msg_id]["explicit"]
            bursts = self._get_bursts(times)
            # print("Delta: {} | Msg id: {} | Num Explicit: {}".format(self.delta_multiplier, msg_id, len(explicit)))
            # print("size(times): {}".format(len(times)))
            # print("size(explicit): {}".format(len(explicit)))
            # print("bursts: {}".format(bursts))

            if len(bursts) > 1:
                print("ERROR MULTIPLE BURSTS DETECTED")
                print(bursts)

            b = bursts[-1] # Only use last burst
            b_numpy = np.array(b)
            start_time = b[0]
            # print("Constructing msg: {}".format(msg_id))
            if len(b) > 1:
                cumdiff = b_numpy[1:] - b_numpy[:-1] # Get the adjacent difference
                latencies = [lat.to_sec() for lat in cumdiff]
                mean_lat = np.mean(latencies)
                # print("Avg latency: {}".format(mean_lat))
            else:
                mean_lat = 0
            # print("Num msgs: {}".format(len(b)))
            burst_msg = self._make_burst_msg(msg_id, len(b), start_time, mean_lat)
            buffer.append( burst_msg )
            explicit_buffer.extend( explicit )
            # report_implicit_count += (len(b) - len(explicit))
        
        meas_sort = lambda x : x.stamp
        explicit_buffer.sort(key=meas_sort, reverse=True)
        buffer.extend(explicit_buffer)

        # REPORT
        # print("******* BUFFER SHARING REPORT FOR {} w/ Delta {}*******".format(self.my_name, self.delta_multiplier))
        # print("Last shared time: {}".format(report_last_shared_time.to_sec()))
        # print("Sharing duration: {}".format(report_duration.to_sec()))
        # print("Sharing time now: {}".format(report_now_last_shared_time.to_sec()))
        # print("Implicit cnt: {}".format(report_implicit_count))
        # print("Explicit cnt: {}".format(len(explicit_buffer)))

        return buffer     # Delta-Tiering
        # return explicit_buffer # N-most recent

    def fillin_buffer(self, buffer, delta_multiplier):
        # Add variances & et-deltas
        buffer = self._add_variances(buffer)
        buffer = self._add_etdeltas(buffer, delta_multiplier)

        # Delta-tiering 
        new_buffer = []       
        implicit_cnt, explicit_cnt = 0, 0

        # N-most recent
        # new_buffer = buffer
        # implicit_cnt, explicit_cnt = 0, len(buffer)
        
        meas_dict = self._get_meas_dict_from_buffer(buffer)
        for msg_id in meas_dict:
            bursts = meas_dict[msg_id]["bursts"]
            explicit = meas_dict[msg_id]["explicit"]
            meas_sort = lambda x : x.stamp
            explicit.sort(key=meas_sort)
            # print("Explicit meas:")
            # print(explicit)
            all_implicit = []
            for b in bursts:
                implicit_meas, avg_latency = self._expand_burst_msg(b)
                all_implicit.extend(implicit_meas)
            all_implicit.sort(key=meas_sort)
            # Match all of the explicit to their corresponding implicit placeholders:
            # This works by first aligning the explicit with the last measurements in the implicit array
            # Then move the first explicit forward in the implicit array to the best match
            # Then proceed with each explicit sequentially, moving it forward and matching to the remaining best fits
            # assert len(all_implicit) >= len(explicit) # Uneeded with following code
            left_over_explicit = []
            if len(all_implicit) < len(explicit): # We dropped a burst msg
                first_implicit, final_implicit = all_implicit[0], all_implicit[-1]
                for m in explicit:
                    if m.stamp < first_implicit.stamp or m.stamp > final_implicit.stamp:
                        left_over_explicit.append( m )
                for m in left_over_explicit:
                    explicit.remove( m )

            size_diff = len(all_implicit) - len(explicit)
            indices = [x + size_diff for x in range(len(explicit))]
            for i in range(len(explicit)):
                start_ind = 0 if i == 0 else indices[i-1]
                end_ind = indices[i]
                if start_ind == end_ind:
                    break
                search_times = [x.stamp for x in all_implicit[start_ind:end_ind] ]
                diffs = [abs( (x - explicit[i].stamp).to_sec() ) for x in search_times]
                best_ind = np.argmin(diffs) + start_ind
                indices[i] = best_ind
            # print("Indices: {}".format(indices))
            for i in range(len(indices)):
                all_implicit[indices[i]] = explicit[i]
            implicit_cnt += (len(all_implicit) - len(indices))
            explicit_cnt += len(indices) + len(left_over_explicit)
            # print(all_implicit)

            new_buffer.extend( all_implicit )
            new_buffer.extend( left_over_explicit )

        return new_buffer
        

    def catch_up(self, start_ind):
        """
        Rewinds the filter and brings it up to the current momment in time
        """
        if self.is_main_filter:
            print("################## Starting Index: {} ################".format(start_ind))

        self.explicit_count = 0

        ledger = deepcopy(self.ledger)
        # print("DT: {}".format(self.delta_multiplier))
        # print(ledger[start_ind]["P_prior"])

        if ledger[start_ind]["x_hat_prior"] is None or ledger[start_ind]["P_prior"] is None:
            start_ind -= 1
            # print("Start index: {}".format(start_ind))

        # Reset the ledger
        self.ledger = {}
        for i in range(1, start_ind):
            self.ledger[i] = ledger[i]
        self._add_block()

        # Reset the filter
        # self.filter = deepcopy(self.original_filter)
        
        self.filter.x_hat = ledger[start_ind]["x_hat_prior"]
        self.filter.P = ledger[start_ind]["P_prior"]
        
        for i_step in range(start_ind,len(ledger)):
            meas_list = ledger[i_step]["meas"]
            update_time = ledger[i_step]["time"]
            u = ledger[i_step]["u"]
            Q = ledger[i_step]["Q"]
            nav_mean = ledger[i_step]["nav_mean"]
            nav_cov = ledger[i_step]["nav_cov"]

            for meas in meas_list:
                self.add_meas(meas)
            self.update(update_time, u, Q, nav_mean, nav_cov)

    def get_asset_estimate(self, asset):
        asset_id = self.asset2id[asset]
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