"""@package etddf

Shares the N most recent measurements of an agent into the buffer

"""
__author__ = "Luke Barbier"
__copyright__ = "Copyright 2020, COHRINT Lab"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"
__license__ = "MIT"
__maintainer__ = "Luke Barbier"

from copy import deepcopy
from etddf.etfilter import ETFilter, ETFilter_Main
from etddf.ros2python import get_internal_meas_from_ros_meas
from etddf_minau.msg import Measurement
import time
from pdb import set_trace as st
import numpy as np
import scipy
import scipy.optimize

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

class MostRecent:
    """Windowed Communication Event Triggered Communication

    Provides a buffer that can be pulled and received from another. Just shares the N most recent measurements of another agent
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
        self.meas_ledger = []
        self.asset2id = asset2id
        self.my_name = my_name

        self.filter = ETFilter(asset2id[my_name], num_ownship_states, 3, x0, P0, True)

        # Remember for instantiating new LedgerFilters
        self.num_ownship_states = num_ownship_states
        self.buffer_capacity = buffer_capacity
        self.meas_space_table = meas_space_table

    def add_meas(self, ros_meas):
        """Adds a measurement to filter

        Arguments:
            ros_meas {etddf.Measurement.msg} -- Measurement taken

        Keyword Arguments:
            delta_multiplier {int} -- not used (left to keep consistent interface)
            force_fuse {bool} -- not used
        """
        src_id = self.asset2id[ros_meas.src_asset]
        if ros_meas.measured_asset in self.asset2id.keys():
            measured_id = self.asset2id[ros_meas.measured_asset]
        elif ros_meas.measured_asset == "":
            measured_id = -1 #self.asset2id["surface"]
        else:
            rospy.logerr("ETDDF doesn't recognize: " + ros_meas.measured_asset + " ... ignoring")
            return
        meas = get_internal_meas_from_ros_meas(ros_meas, src_id, measured_id, 1)
        self.filter.add_meas(meas)
        self.meas_ledger.append(ros_meas)

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
        """Runs covariance intersection with filter's estimate

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
        
        c_bar, Pcc = DeltaTier.run_covariance_intersection(x, P, x_prior, P_prior)

        # Update main filter states
        if Pcc.shape != self.filter.P.shape:
            # self.psci(x_prior, P_prior, c_bar, Pcc)
            self.filter.x_hat[begin_ind:end_ind] = c_bar
            self.filter.P[begin_ind:end_ind,begin_ind:end_ind] = P_prior
        else:
            self.filter.x_hat = c_bar
            self.filter.P = Pcc

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
            Updates self.filter.x_hat and P, the delta tier's primary estimate
        """
        x = self.filter.x_hat
        P = self.filter.P

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

        self.filter.x_hat = posterior_state
        self.filter.P = posterior_cov


    def catch_up(self, delta_multiplier, shared_buffer, Q):
        """Updates estimate based on buffer

        Arguments:
            delta_multiplier {float} -- multiplier to scale et_delta's with
            shared_buffer {list} -- buffer shared from another asset
        Returns:
            int -- implicit measurement count in shared_buffer
            int -- explicit measurement count in this shared_buffer
        """
        for meas in shared_buffer: # Fuse all of the measurements now
            self.add_meas(meas)

    def pull_buffer(self):
        """Pulls all measurements that'll fit

        Returns:
            multiplier {float} -- the delta multiplier that was chosen
            buffer {list} -- the buffer of ros measurements
        """
        buffer = []
        cost = 0
        ind = -1
        while abs(ind) <= len(self.meas_ledger):
            new_meas = self.meas_ledger[ind]
            space = self.meas_space_table[new_meas.meas_type]
            if cost + space <= self.buffer_capacity:
                buffer.append(new_meas)
                cost += space
            ind -= 1
        self.meas_ledger = []
        return 1, buffer

    def predict(self, u, Q, time_delta=1.0, use_control_input=False):
        """Executes prediction step on all filters

        Arguments:
            u {np.ndarray} -- my asset's control input (num_ownship_states / 2, 1)
            Q {np.ndarray} -- motion/process noise (nstates, nstates)

        Keyword Arguments:
            time_delta {float} -- Amount of time to predict in future (default: {1.0})
            use_control_input {bool} -- Use control input on filter
        """
        self.filter.predict(u, Q, time_delta, use_control_input=use_control_input)

    def correct(self, update_time):
        """Executes correction step on all filters

        Arguments:
            update_time {time} -- not used (kept for interface)

        """
        self.filter.correct()

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
        asset_mean = self.filter.x_hat[begin_ind:end_ind,0]
        asset_unc = self.filter.P[begin_ind:end_ind,begin_ind:end_ind]
        return deepcopy(asset_mean), deepcopy(asset_unc)

    def debug_print_buffers(self):
        return self.meas_ledger

if __name__ == "__main__":

    # TODO to run these tests, uncomment Measurement class above

    # Test plumbing
    test_buffer_pull = True
    test_catch_up = not test_buffer_pull

    import numpy as np
    x0 = np.zeros((6,1))
    P = np.ones((6,6)) * 10
    meas_space_table = {"depth":2, "dvl_x":2,"dvl_y":2, "bookend":1,"bookstart":1, "final_time":0}
    delta_codebook_table = {"depth":1.0, "dvl_x":1, "dvl_y":1}
    asset2id = {"my_name":0}
    buffer_cap = 10
    dt = MostRecent(6, x0, P, buffer_cap, meas_space_table, 0, delta_codebook_table, [0.5,1.5], asset2id, "my_name")

    Q = np.eye(6); Q[3:,3:] = np.zeros(Q[3:,3:].shape)
    u = np.array([[0.1,0.1,-0.1]]).T
    t1 = time.time()
    z = Measurement("depth", t1, "my_name","", -1, 0.1, [])
    dvl_x = Measurement("dvl_x", t1, "my_name","", 1, 0.1, [])
    dvl_y = Measurement("dvl_y", t1, "my_name","", 1, 0.1, [])

    dt.add_meas(z)
    dt.add_meas(dvl_x)
    dt.add_meas(dvl_y)
    dt.predict(u,Q)
    dt.correct(t1)

    t2 = time.time()
    z.stamp = t2
    dvl_x.stamp = t2
    dvl_y.stamp = t2
    dvl_x.data = 2
    dvl_y.data = 2
    dt.add_meas(z)
    dt.add_meas(dvl_x)
    dt.add_meas(dvl_y)
    dt.predict(u,Q)
    dt.correct(t2)
    print(dt.get_asset_estimate("my_name"))

    ##### Test Buffer Pulling #####
    if test_buffer_pull:
        mult, buffer = dt.pull_buffer()
        assert len(buffer) == 5
        strbuffer = [x.meas_type for x in buffer]
        print(strbuffer)
        print("Should be empty:")
        print(dt.debug_print_buffers())

    ##### Test catching up #####
    if test_catch_up:
        print("{:.20f}".format(t1))
        print("{:.20f}".format(t2))
        mult, buffer = dt.pull_buffer()
        buf_contents = [x.meas_type for x in buffer]
        print(buf_contents)

        dt2 = MostRecent(6, x0, P, buffer_cap, meas_space_table, 0, delta_codebook_table, [0.5,1.5], asset2id, "my_name")
        dt2.predict(u, Q)
        dt2.correct(t1)
        dt2.predict(u, Q)

        print("catch up")
        dt2.catch_up(mult, buffer)
        dt2.correct(t2)
        print(dt2.get_asset_estimate("my_name"))