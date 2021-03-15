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
import scipy
import scipy.optimize

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
            missed_meas_tolerance_table {dict} -- Hash that determines how many measurements of each type do we need to miss before indicating a bookend
            delta_codebook_table {dict} -- Hash that stores delta trigger for each measurement type. Str(meas type) -> float (delta trigger)
            delta_multipliers {list} -- List of delta trigger multipliers
            asset2id {dict} -- Hash to get the id number of an asset from the string name
            my_name {str} -- Name to loopkup in asset2id the current asset's ID#
        """
        self.ledger_update_times = []
        self.asset2id = asset2id
        self.my_name = my_name

        # Initialize Common Ledger Filters
        self.delta_tiers = {}
        for multiplier in delta_multipliers:
            self.delta_tiers[multiplier] = LedgerFilter(
                num_ownship_states, x0, P0, \
                buffer_capacity, meas_space_table, \
                missed_meas_tolerance_table, \
                delta_codebook_table, multiplier, \
                False, asset2id[my_name]
            )

        # Initialize Main Filter
        self.main_filter = LedgerFilter(
            num_ownship_states, x0, P0, \
            buffer_capacity, meas_space_table, \
            missed_meas_tolerance_table, \
            delta_codebook_table, 69, \
            True, asset2id[my_name]
        ) # Make delta multiplier high so we don't share meas using this filter

        # Remember for instantiating new LedgerFilters
        self.num_ownship_states = num_ownship_states
        self.buffer_capacity = buffer_capacity
        self.meas_space_table = meas_space_table
        self.missed_meas_tolerance_table = missed_meas_tolerance_table
        self.delta_codebook_table = delta_codebook_table
        self.delta_multipliers = delta_multipliers

    def add_meas(self, ros_meas, delta_multiplier=USE_DELTA_TIERS, force_fuse=True):
        """Adds a measurement to all ledger filters.
        
        If Measurement is after last correction step, it will be fused on next correction step

        Arguments:
            ros_meas {etddf.Measurement.msg} -- Measurement taken

        Keyword Arguments:
            delta_multiplier {int} -- Delta multiplier to use for this measurement (default: {USE_DELTA_TIERS})
            force_fuse {bool} -- If measurement is in the past, fuse it on the next update step anyway (default: {True})
                Note: the ledger will still reflect the correct measurement time
        """
        ros_meas = deepcopy(ros_meas)
        src_id = self.asset2id[ros_meas.src_asset]
        if ros_meas.measured_asset in self.asset2id.keys():
            measured_id = self.asset2id[ros_meas.measured_asset]
        elif ros_meas.measured_asset == "":
            measured_id = -1 #self.asset2id["surface"]
        else:
            print("!!!!!!!!! ETDDF doesn't recognize: " + ros_meas.measured_asset + " ... ignoring")
            return
        self.main_filter.add_meas(ros_meas, src_id, measured_id)
        for key in self.delta_tiers.keys():
            self.delta_tiers[key].add_meas(ros_meas, src_id, measured_id)

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
        x_prior = self.main_filter.filter.x_hat[begin_ind:end_ind].reshape(-1,1)
        P_prior = self.main_filter.filter.P[begin_ind:end_ind,begin_ind:end_ind]
        P_prior = P_prior.reshape(self.num_ownship_states, self.num_ownship_states)
        
        c_bar, Pcc = DeltaTier.run_covariance_intersection(x, P, x_prior, P_prior)

        # Update main filter states
        if Pcc.shape != self.main_filter.filter.P.shape:
            # self.psci(x_prior, P_prior, c_bar, Pcc)
            self.main_filter.filter.x_hat[begin_ind:end_ind] = c_bar
            self.main_filter.filter.P[begin_ind:end_ind,begin_ind:end_ind] = P_prior
        else:
            self.main_filter.filter.x_hat = c_bar
            self.main_filter.filter.P = Pcc

        return c_bar, Pcc

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
        x = self.main_filter.filter.x_hat
        P = self.main_filter.filter.P

        D_inv = np.linalg.inv(P_prior) - np.linalg.inv(Pcc)
        D_inv_d = np.dot( np.linalg.inv(P_prior), x_prior) - np.dot( np.linalg.inv(Pcc), c_bar)
        
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

        self.main_filter.filter.x_hat = posterior_state
        self.main_filter.filter.P = posterior_cov


    def catch_up(self, delta_multiplier, shared_buffer, Q):
        """Updates main estimate and common estimate using the shared buffer

        Arguments:
            delta_multiplier {float} -- multiplier to scale et_delta's with
            shared_buffer {list} -- buffer shared from another asset
            Q {np.ndarray} -- motion/process noise (nstates, nstates)
        """
        # Fill in implicit measurements in the buffer and align the meas timestamps with our own
        # print("--- catching up ---")
        new_buffer_names = [x.meas_type for x in shared_buffer]
        # print(new_buffer_names)
        new_buffer = self._fillin_buffer(shared_buffer)
        new_buffer_names2 = [x.meas_type for x in new_buffer]
        # print(new_buffer_names2)

        # Mark all measurement ledgers with triggers
        for i in range(len(new_buffer)):
            new_buffer[i].et_delta = delta_multiplier
        for i in range(len(self.main_filter.ledger_meas)):
            self.main_filter.ledger_meas[i].et_delta = 1.0
        for mult in self.delta_tiers:
            for i in range(len(self.delta_tiers[mult].ledger_meas)):
                self.delta_tiers[mult].ledger_meas[i].et_delta = mult

        meas_ledgers = {}
        # Merge new buffer with our measurements
        sortfxn = lambda x : x.stamp
        for mult in self.delta_tiers:
            meas_ledgers[mult] = self.delta_tiers[mult].ledger_meas + new_buffer
            meas_ledgers[mult].sort(key=sortfxn)
            new_buffer_names = [x.meas_type for x in self.delta_tiers[mult].ledger_meas]
            # print(new_buffer_names)
            new_buffer_names = [x.meas_type for x in meas_ledgers[mult]]
            # print(new_buffer_names)
            # print("---")

        main_ledger = self.main_filter.ledger_meas + new_buffer
        main_ledger.sort(key=sortfxn)
        # new_buffer_names = [x.meas_type for x in meas_ledgers[mult]]
        update_times = self.main_filter.ledger_update_times

        main_x0, main_P0 = self.main_filter.original_estimate
        comm_x0, comm_P0 = list(self.delta_tiers.values())[0].original_estimate

        # Initialize common filters
        my_id = self.asset2id[self.my_name]
        delta_tiers = {}
        for mult in self.delta_tiers:
            delta_tiers[mult] = ETFilter(my_id, self.num_ownship_states, 3, comm_x0, comm_P0, True)
        main_filter = ETFilter(my_id, self.num_ownship_states, 3, main_x0, main_P0, True)


        # Local function to simplify loops below
        def ros2pythonMeas(ros_meas):
            # Convert the ros meas to python measurement (bit of a pain)
            meas_type = meas.meas_type.split("_implicit")[0]
            delta = self.delta_codebook_table[meas_type]
            src_id = self.asset2id[meas.src_asset]
            if meas.measured_asset in list(self.asset2id):
                measured_id = self.asset2id[meas.measured_asset]
            else:
                measured_id = -1
            return get_internal_meas_from_ros_meas(meas, src_id, measured_id, delta * ros_meas.et_delta)

        # Catch up common filters
        update_ind = 0
        for mult in self.delta_tiers:
            for meas in meas_ledgers[mult]:
                if meas.stamp > update_times[update_ind]:
                    delta_tiers[mult].predict(np.zeros((3,1)), Q)
                    delta_tiers[mult].correct()
                    update_ind += 1
                    if update_ind >= len(update_times): # Ignore measurements in the future
                        break

                    # print("updating..")
                delta_tiers[mult].add_meas( ros2pythonMeas( meas ) )
            update_ind = 0
        for mult in self.delta_tiers: # Update once more
            delta_tiers[mult].predict(np.zeros((3,1)), Q)
            delta_tiers[mult].correct()
        # Catch up main filter
        update_ind = 0
        for meas in main_ledger:
            if meas.stamp > update_times[update_ind]:
                main_filter.predict(np.zeros((3,1)), Q)
                main_filter.correct()
                update_ind += 1
                if update_ind >= len(update_times): # Ignore measurements in the future
                    break
                # print("updating..")
            main_filter.add_meas( ros2pythonMeas( meas ) )
        main_filter.predict(np.zeros((3,1)), Q) # Update once more
        main_filter.correct()

        new_main_xhat = main_filter.x_hat
        new_main_P = main_filter.P
        old_main_xhat = self.main_filter.filter.x_hat
        old_main_P = self.main_filter.filter.P
        c_bar, Pcc = DeltaTier.run_covariance_intersection(
            new_main_xhat, new_main_P,
            old_main_xhat, old_main_P)
        # Update the main estimates
        main_filter.x_hat = c_bar
        main_filter.P = Pcc
        
        # Update the Filters
        for mult in self.delta_tiers:
            self.delta_tiers[mult].filter = delta_tiers[mult]
        self.main_filter.filter = main_filter

        #### MEASUREMENT LEDGER ####
        new_buffer.sort(key=sortfxn)
        first_meas = new_buffer[0]
        for mult in self.delta_tiers:
            ledger = meas_ledgers[mult]#self.delta_tiers[mult].ledger_meas
            trim = 0
            for i in range(len(ledger)):
                if ledger[i].stamp < first_meas.stamp :
                    trim = i
            self.delta_tiers[mult].ledger_meas = deepcopy(ledger[trim:])
        ledger = main_ledger
        trim = 0
        for i in range(len(ledger)):
            if ledger[i].stamp < first_meas.stamp :
                trim = i
        # if trim != 0:
        #     print("Trimming Meas Ledger")
        self.main_filter.ledger_meas = deepcopy(ledger[trim:])

        #### UPDATE TIMES ####
        update_times = self.main_filter.ledger_update_times
        trim = 0
        for i in range(len(update_times)):
            if update_times[i] < first_meas.stamp:
                trim = i
        # if trim != 0:
        #     print("Trimming update times")
        self.main_filter.ledger_update_times = deepcopy(update_times[trim:])
        for mult in self.delta_tiers:
            self.delta_tiers[mult].ledger_update_times = deepcopy(update_times[trim:])

    def peek_buffer(self):
        """ Get the current lowest multiplier buffer without clearing it

        Returns:
            bool -- whether the buffer has been overflown
            float -- the lowest delta tier multiplier
            list -- the buffer of the lowest not overflown buffer
        """
        # Find lowest delta tier that hasn't overflown
        lowest_multiplier = -1
        for key in self.delta_tiers:
            if not self.delta_tiers[key].check_overflown():
                lowest_multiplier = key
                break
        if lowest_multiplier == -1:
            return True, -1.0, []
        else:
            return False, lowest_multiplier, self.delta_tiers[lowest_multiplier].peek()

    def pull_buffer(self):
        """Pulls lowest delta multiplier's buffer that hasn't overflown

        Returns:
            multiplier {float} -- the delta multiplier that was chosen
            buffer {list} -- the buffer of ros measurements
        """
        # Find lowest delta tier that hasn't overflown
        not_overflown_list = []
        for key in list(self.delta_tiers.keys()):
            if not self.delta_tiers[key].check_overflown():
                not_overflown_list.append(key)
            
        # TODO add overflow support -> possibly at runtime rather than extend this callback
        if not not_overflown_list:
            raise NotImplementedError("All deltatier buffers have overflown")
        else:
            lowest_multiplier = min(not_overflown_list)

        last_time = self.main_filter.ledger_update_times[-1]
        buffer = self.delta_tiers[lowest_multiplier].flush_buffer(last_time)
        
        # Change old delta tiers to be copies of the selected one
        for key in list(self.delta_tiers.keys()):
            if key != lowest_multiplier:
                del self.delta_tiers[key]
                self.delta_tiers[key] = deepcopy(self.delta_tiers[lowest_multiplier])
                self.delta_tiers[key].convert(key) # Change the delta multiplier to the new one
                
        return lowest_multiplier, buffer

    def predict(self, u, Q, time_delta=1.0, use_control_input=False):
        """Executes prediction step on all filters

        Arguments:
            u {np.ndarray} -- my asset's control input (num_ownship_states / 2, 1)
            Q {np.ndarray} -- motion/process noise (nstates, nstates)

        Keyword Arguments:
            time_delta {float} -- Amount of time to predict in future (default: {1.0})
            use_control_input {bool} -- Use control input on main filter
        """
        self.main_filter.predict(u, Q, time_delta, use_control_input=use_control_input)
        for key in self.delta_tiers.keys():
            self.delta_tiers[key].predict(u, Q, time_delta, use_control_input=False)

    def correct(self, update_time):
        """Executes correction step on all filters

        Arguments:
            update_time {time} -- Update time to record on the ledger update times

        Raises:
            ValueError: Update time is before the previous update time

        """
        if (len(self.main_filter.ledger_update_times) > 1) and (update_time <= self.main_filter.ledger_update_times[-1]):
            raise ValueError("update time must be later in time than last update time")

        self.main_filter.correct(update_time)
        for key in self.delta_tiers.keys():
            self.delta_tiers[key].correct(update_time)

    def get_asset_estimate(self, asset_name):
        """Gets main filter's estimate of an asset

        Arguments:
            asset_name {str} -- Name of asset

        Returns
            np.ndarray -- Mean estimate of asset (num_ownship_states, 1)
            np.ndarray -- Covariance of estimate of asset (num_ownship_states, num_ownship_states)
        """
        asset_id = self.asset2id[asset_name]
        begin_ind = asset_id*self.num_ownship_states
        end_ind = (asset_id+1)*self.num_ownship_states
        asset_mean = self.main_filter.filter.x_hat[begin_ind:end_ind,0]
        asset_unc = self.main_filter.filter.P[begin_ind:end_ind,begin_ind:end_ind]
        return deepcopy(asset_mean), deepcopy(asset_unc)

    def _fillin_buffer(self, shared_buffer):
        """Fills in implicit measurements to buffer and generates time indices

        Arguments:
            shared_buffer {list} -- List of measurements received from other asset

        Returns:
            list -- filled in list of measurements
            int -- last ledger time index
        """
        # Sort the buffer cronologically
        fxn = lambda x : x.stamp
        shared_buffer.sort(key=fxn)
        # print([x.meas_type for x in shared_buffer])

        new_buffer = []
        expected_meas = {}

        get_marker = lambda meas_type, measured_asset : meas_type + "_" + measured_asset

        update_ind = 0
        for meas in shared_buffer:
            if meas.stamp > self.main_filter.ledger_update_times[update_ind]:
                # print("updating index...")
                
                # Somehow we're future in time, shouldn't be possible...
                if update_ind == len(self.main_filter.ledger_update_times): 
                    meas.stamp = self.main_filter.ledger_update_times[-1]
                else:

                    update_ind += 1
                    # Add implicit measurements
                    # print(list(expected_meas.keys()))
                    # print([x.meas_type for x in new_buffer])
                    for marker in expected_meas:
                        expected_meas[marker].stamp = self.main_filter.ledger_update_times[update_ind]
                        if "implicit" not in expected_meas[marker].meas_type:
                            expected_meas[marker].meas_type += "_implicit"
                            continue
                        new_buffer.append(deepcopy(expected_meas[marker]))
                

            if "bookstart" in meas.meas_type:
                meas_type = meas.meas_type.split("_bookstart")[0]
                meas.meas_type = meas_type + "_implicit"
                expected_meas[get_marker(meas_type, meas.measured_asset)] = deepcopy(meas)
                continue
            elif "bookend" in meas.meas_type:
                meas_type = meas.meas_type.split("_bookend")[0]
                del expected_meas[get_marker(meas_type, meas.measured_asset)]
                continue
            elif meas.meas_type == "final_time":
                continue
            else:
                expected_meas[get_marker(meas.meas_type, meas.measured_asset)] = deepcopy(meas)
            new_buffer.append(meas)

        # print(list(expected_meas.keys()))
        # print([x.meas_type for x in new_buffer])
        for marker in expected_meas:
            expected_meas[marker].stamp = self.main_filter.ledger_update_times[update_ind]
            if "implicit" not in expected_meas[marker].meas_type:
                expected_meas[marker].meas_type += "_implicit"
                continue
            new_buffer.append(deepcopy(expected_meas[marker]))

        new_buffer.sort(key=fxn)
        # print([x.meas_type for x in new_buffer])

        return new_buffer

    def debug_print_meas_ledgers(self, multiplier):
        """Prints the measurement ledger of the multiplier's filter

        Arguments:
            multiplier {float} -- Must be a key in self.delta_tiers
        """
        measurements = [x.meas_type for x in self.delta_tiers[multiplier].ledger_meas]
        print(measurements)

    def debug_common_estimate(self, multiplier):
        """Returns the estimate of the deltatier filter

        Arguments:
            multiplier {float} -- Must be a key in self.delta_tiers
                If 'None', returns the main filter estimate

        Returns:
            np.array -- Mean of estimate of the filter
            np.array -- Covariance of estimate of the filter
        """
        if multiplier is None:
            return deepcopy(self.main_filter.filter.x_hat), deepcopy(self.main_filter.filter.P)
        else:
            return deepcopy(self.delta_tiers[multiplier].filter.x_hat), deepcopy(self.delta_tiers[multiplier].filter.P)

    def debug_print_buffers(self):
        """Prints the contents of all of the delta tier buffers
        """
        for dt_key in self.delta_tiers.keys():
            buffer = [x.meas_type for x in self.delta_tiers[dt_key].buffer.buffer]
            if self.delta_tiers[dt_key].check_overflown():
                print(str(dt_key) + " is overflown")
            print(str(dt_key) + " " + str(buffer))

if __name__ == "__main__":

    # TODO to run these tests, uncomment Measurement class in ledger_filter.py


    # Test plumbing
    import numpy as np
    import sys
    from etddf.ledger_filter import Measurement

    x0 = np.zeros((6,1))
    P = np.eye(6); P[3:,3:] *= 0.1
    meas_space_table = {"depth":2, "dvl_x":2,"dvl_y":2, "bookend":1,"bookstart":1, "final_time":0}
    delta_codebook_table = {"depth":1.0, "dvl_x":1, "dvl_y":1}
    missed_meas_tolerance_table = {"depth":1, "dvl_x":1, "dvl_y":1}
    asset2id = {"my_name":0}
    buffer_cap = 8
    dt = DeltaTier(6, x0, P, buffer_cap, meas_space_table, missed_meas_tolerance_table, delta_codebook_table, [0.5,1.5], asset2id, "my_name")

    Q = np.eye(6); Q[3:,3:] *= 0.001
    # u = np.array([[0.1,0.1,-0.1]]).T
    u = np.array([[0.0, 0, 0]]).T
    t1 = time.time()
    dvl_x = Measurement("dvl_x", t1, "my_name","", 1, 0.1, [])
    dvl_y = Measurement("dvl_y", t1, "my_name","", 1, 0.1, [])

    test_normal, test_buffer_pull, test_catch_up = True, False, True

    ##### Test Normal Delta Tiering: bookstarts, bookends
    dt.add_meas(dvl_x)
    dt.add_meas(dvl_y)
    dt.predict(u,Q)
    dt.correct(t1)
    if test_normal:
        print("0.5 should have dvl_x,dvl_y")
        print("1.5 should have bookstarts for dvl_x,dvl_y")
        dt.debug_print_buffers()
        print("---")
    dvl_x1 = deepcopy(dvl_x)

    t2 = time.time()
    dvl_x.stamp = t2
    dvl_y.stamp = t2
    dvl_x.data = 1
    dvl_y.data = 2
    dt.add_meas(dvl_x)
    dt.add_meas(dvl_y)
    dt.predict(u,Q)
    dt.correct(t2)
    if test_normal:
        print("0.5 should have dvl_x,dvl_y, dvl_y")
        print("1.5 should have bookstarts{dvl_x,dvl_y}, dvl_y")
        dt.debug_print_buffers()
        print("---")
    dvl_x2 = deepcopy(dvl_x)
    

    t3 = time.time()
    dvl_x.stamp = t3
    dvl_y.stamp = t3
    # dt.add_meas(dvl_x)
    dt.add_meas(dvl_y)
    dt.predict(u,Q)
    dt.correct(t3)
    if test_normal:
        print("0.5 should be overflown")
        print("1.5 should have bookstarts{dvl_x,dvl_y}, dvl_y,  dvl_x_bookend")
        dt.debug_print_buffers()
        print("---")

    t4 = time.time()
    dvl_y.stamp = t4
    # dt.add_meas(dvl_x)
    dt.add_meas(dvl_y)
    dt.predict(u,Q)
    dt.correct(t4)
    if test_normal:
        print("0.5 should be overflown")
        print("1.5 should have bookstarts{dvl_x,dvl_y}, dvl_x, dvl_y,  dvl_x_bookend")
        dt.debug_print_buffers()
        print("---")

    ##### Test Buffer Pulling #####
    if test_buffer_pull:
        print("0.5 should be overflown")
        print("1.5 should have bookstarts{dvl_x,dvl_y}, dvl_y, dvl_x_bookend")
        print("Our buffer should be the 1.5 one")
        dt.debug_print_buffers()
        mult, buffer = dt.pull_buffer()
        strbuffer = [x.meas_type for x in buffer]
        print(strbuffer)
        print("Both should be empty")
        dt.debug_print_buffers()

    ##### Test catching up #####
    if test_catch_up:
        print("meas ledgers of: " + str(1.5))
        # dt.debug_print_meas_ledgers(1.5)
        # dt.debug_print_buffers()
        mult, buffer = dt.pull_buffer()
        print(mult)
        buf_contents = [x.meas_type for x in buffer]
        # print(buf_contents)
        from random import shuffle
        shuffle(buffer)

        # strbuffer = [x.meas_type for x in buffer]
        # print(strbuffer)

        dt2 = DeltaTier(6, x0, P, buffer_cap, meas_space_table, missed_meas_tolerance_table, delta_codebook_table, [0.5,1.5], asset2id, "my_name")
        # dt2.add_meas(dvl_x1)
        dt2.predict(u, Q)
        dt2.correct(t1)
        # dt2.add_meas(dvl_x2)
        dt2.predict(u, Q)
        dt2.correct(t2)
        # dt2.add_meas(dvl_x)
        dt2.predict(u, Q)
        dt2.correct(t3)

        dt2.predict(u, Q)
        dt2.correct(t4)

        # print(dt.main_filter.filter.x_hat)
        # print(dt.main_filter.filter.P)
        print("catching up")
        dt2.catch_up(mult, buffer, Q)
        # print(dt2.main_filter.filter.x_hat)
        # print(dt2.main_filter.filter.P)
        for mult in dt2.delta_tiers:
            print(dt.delta_tiers[mult].filter.x_hat)
            print(dt2.delta_tiers[mult].filter.x_hat)
            print(dt.delta_tiers[mult].filter.P)
            print(dt2.delta_tiers[mult].filter.P)
            print("-------")

        print("Thesee should be same same")
        dt.debug_print_meas_ledgers(1.5)
        dt2.debug_print_meas_ledgers(1.5)
        print("###")

        print("These shouldn't be the same. One has had explicit dvl_x")
        x_hat, P = dt.debug_common_estimate(None)
        x_hat2, P2 = dt2.debug_common_estimate(None)
        print(x_hat)
        print(x_hat2)

        # Should see the same meas ledgers as above for both
        # dt2.debug_print_meas_ledgers(0.5)
        # dt2.debug_print_meas_ledgers(1.5)