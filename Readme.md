Trial code for DLprof on Nvidia GPUs and hardware

# Installation
`
python3 -m pip install nvidia-pyindex
python3 -m pip install nvidia-dlprof
`

# Run for pytorch dummy network
`
dlprof --mode=pytorch --force true python3 dummy_network.py
dlprof --mode=pytorch --database=nsys_profile.sqlite
`
# Reports Generation
`
dlprof --mode=pytorch --reports=all --database=nsys_profile.sqlite 
`
# Reference
`
https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/ 
`