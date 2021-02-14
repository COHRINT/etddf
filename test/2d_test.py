#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import norm as normal
from etddf.asset import Asset
from etddf.measurements import *
from copy import deepcopy
from error_plotting import *
from etddf.dynamics import *

DEBUG=False

# Simple simulation
K = 1000
world_dim = 1
num_assets = 3
num_ownship_states = 2
num_states = num_ownship_states * num_assets
linear_dynamics_status = True

def main():

    ######## DEFINE STATE && UNCERTAINTY #############
    x_truth = np.array([[0,0,5,0,7,0]], dtype=np.float64).T
    P_initial = np.array([[1,0,0,0,0,0], \
                          [0,4,0,0,0,0], \
                          [0,0,1,0,0,0], \
                          [0,0,0,4,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,0,0,4]], dtype=np.float64)

    ####### DEFINE PROCESS NOISE #######
    q = 0.1

    ####### DEFINE PERCEIVED PROCESS NOISE #######
    Q_perceived = np.array([[4,0,0,0,0,0], \
                            [0,0,0,0,0,0], \
                            [0,0,4,0,0,0], \
                            [0,0,0,0,0,0],
                            [0,0,0,0,4,0],
                            [0,0,0,0,0,0]], dtype=np.float64)


    ########## DEFINE MEAS NOISE ##########
    r_gps = 0.1
    r_gps_perceived = 1

    ########## DEFINE ET DELTAS ##########
    gps_xy_delta = 1.0

    ########## INITIALIZE ASSETS ##########
    asset_list = []
    asset = Asset(0, num_ownship_states, world_dim, x_truth, P_initial, linear_dynamics_status, red_team=[2])
    asset1 = Asset(1, num_ownship_states, world_dim, x_truth, P_initial, linear_dynamics_status, red_team=[2])
    asset_list.append(asset); asset_list.append(asset1)

    ########## DATA RECORDING ##########
    x_truth_bag = x_truth_bag = deepcopy(x_truth).reshape(-1,1)
    x_hat_bag0 = deepcopy(asset.main_filter.x_hat).reshape(-1,1)
    p_bag0 = deepcopy(asset.main_filter.P)
    x_hat_bag1 = deepcopy(asset1.main_filter.x_hat).reshape(-1,1)
    p_bag1 = deepcopy(asset1.main_filter.P)
    
    seq = 0
    implicit_update_cnt = 0
    total_num_meas_cnt = 0

    for k in range(K):
        ########## DEFINE CONTROL INPUT ##########
        u0 = np.array([0.2], dtype=np.float64).reshape(-1,1)
        u1 = np.array([0.1], dtype=np.float64).reshape(-1,1)
        u2 = np.array([0.1], dtype=np.float64).reshape(-1,1)

        ########## SIMULATE TRUTH MOTION ##########
        # x_truth_prev = deepcopy(x_truth)
        x_truth0 = linear_propagation(x_truth[:num_ownship_states,0], u0, world_dim, num_ownship_states)[0].reshape(-1,1)
        x_truth1 = linear_propagation(x_truth[num_ownship_states:2*num_ownship_states,0], u1, world_dim, num_ownship_states)[0]
        x_truth2 = linear_propagation(x_truth[2*num_ownship_states:3*num_ownship_states,0], u2, world_dim, num_ownship_states)[0]
        x_truth[:num_ownship_states,0] = x_truth0.ravel()
        x_truth[num_ownship_states:2*num_ownship_states,0] = x_truth1.ravel()
        x_truth[2*num_ownship_states:3*num_ownship_states,0] = x_truth2.ravel()

        ########## BAG TRUTH DATA ##########
        x_truth_bag = np.concatenate((x_truth_bag, x_truth.reshape(-1,1)), axis=1)

        ########## ADD NOISE TO TRUTH MOTION ##########
        x_truth_no_noise = deepcopy(x_truth)
        x_truth[0,0] += np.random.normal(0, np.sqrt(q))
        x_truth[num_ownship_states,0] += + np.random.normal(0, np.sqrt(q))
        x_truth[2*num_ownship_states,0] += + np.random.normal(0, np.sqrt(q))

        ########## PREDICTION STEP  ##########
        asset.predict(u0, Q_perceived)
        asset1.predict(u1, Q_perceived)

        # continue #** CHECK

        ########## GENERATE MEASUREMENTS VALUES ##########
        gpsx0 = x_truth[0,0]
        gpsx1 = x_truth[num_ownship_states,0]
        gpsx2 = x_truth[2*num_ownship_states,0]        

        ########## ADD NOIES TO MEASUREMENTS ##########
        gpsx0 += np.random.normal(0, r_gps)
        gpsx1 += np.random.normal(0, r_gps)
        gpsx2 += np.random.normal(0, r_gps)

        # STOP, CHECK PERFECT MEASUREMENTS
    
        ########## INITIALIZE MEASUREMENT TYPES ##########
        gpsx0_meas = GPSx_Explicit(0, gpsx0, r_gps_perceived**2, gps_xy_delta)
        gpsx1_meas = GPSx_Explicit(1, gpsx1, r_gps_perceived**2, gps_xy_delta)
        gpsx2_meas = GPSx_Neighbor_Explicit(0,2, gpsx2, r_gps_perceived**2, gps_xy_delta)

        ########## ASSETS SHARE MEASUREMENTS  ##########
        to_share1 = asset.receive_meas(gpsx0_meas, shareable=True)
        to_share2 = asset.receive_meas(gpsx2_meas, shareable=True)
        to_share3 = asset1.receive_meas(gpsx1_meas, shareable=True)
        sharing = []
        sharing.append( to_share1 )
        sharing.append( to_share2 )
        sharing.append( to_share3 )
        
        # Share measurements
        for s in sharing:
            i = list(s.keys())[0]
            if isinstance(s[i], Implicit):
                implicit_update_cnt += 1
            total_num_meas_cnt += 1
            a = asset_list[i]
            meas = s[i]
            a.receive_shared_meas(meas)

        ########## CORRECTION STEP ##########
        asset.correct()
        asset1.correct()

        ########## RECORD FITLER DATA ##########

        # Estimate 0
        x_hat_bag0 = np.concatenate((x_hat_bag0, asset.main_filter.x_hat), axis=1)
        p_bag0 = np.concatenate((p_bag0, asset.main_filter.P), axis=1)
        x_hat_bag1 = np.concatenate((x_hat_bag1, asset1.main_filter.x_hat), axis=1)
        p_bag1 = np.concatenate((p_bag1, asset1.main_filter.P), axis=1)
            

        ########## DEBUG FILTER INPUTS ##########
        if DEBUG:
            print(asset.main_filter.P)
            print("---")
            # set_trace()

        seq += 1
        print(str(seq) + " out of " + str(K))

    # STEP CHECK MEAN ESTIMATES ARE DECENT

    print("Percent of msgs sent implicitly: " + str((implicit_update_cnt / total_num_meas_cnt)*100))
    # PLOT ERROR BOUNDS
    plot_error(x_truth_bag, x_hat_bag0, p_bag0, num_ownship_states, 0)
    plot_error(x_truth_bag, x_hat_bag1, p_bag1, num_ownship_states, 1)

def normalize_angle(angle):
    return np.mod( angle + np.pi, 2*np.pi) - np.pi

if __name__ == "__main__":
    main()