# etddf
Event Triggered Distributed Data Fusion

## Setup
```
virtualenv -p python3 etddf-env
source etddf-env/bin/activate
pip install -r requirements.txt
python setup.py install
```

## 2D Test case
Run a 2D test case with two (blue) agents event triggering measurements of themselves and of one target (red) agent.
```
cd test
python 2d_test.py
```
This should run 1000 steps of a simulation and open up two plots when done. The first plot will be the first agent's estimates of all variables. The second plot will be the same for the 2nd blue agent. The terminal should also print out what percentage of measurements were sent implicitly. By increasing the __gps_xy_delta__ variable, this percentage will increase.