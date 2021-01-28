# Parallel Batch RCM 

This repository provides the code to the paper:  
**Speculative Parallel Reverse Cuthill-McKee Reordering on Multi- and Many-core Architectures**  
*Daniel Mlakar, Martin Winter, Mathias Parger, Markus Steinberger*  

## Compilation
```
mkdir build && cd build  
cmake -DCUDA_BUILD_CC=<YOUR_COMPUTE_CAPABILITY>, e.g., 70> ..  
make  
```

NOTE: Supported values for `YOUR_COMPUTE_CAPABILITY` are `61`, `70` and `75`. If none specified the project is built for all supported compute capabilities

## Usage
```
  ./CuthillMcKee input {OPTIONS}  

    Computes the Cuthill-McKee reordering of a sparse quadratic matrix given in  
    the CSR format.

  OPTIONS:  
      -h, --help                        Display this help menu  
      input                             quadratic matrix in CSR format to reorder  
      -i[implementation],
      --implementation=[implementation] Select an RCM implementation. Available values are: {ALL, cuSolverRCM, CPU, CPU_BATCH, GPU, GPU_BATCH}  
      -r, --reverse                     Revert CM permutation (RCM).  
      -s[start], --start=[start]        Manually pick start node. If not specified, a pseudo-peripheral node is selected automatically.  
      -b[stable], --stable=[stable]     Select whether sorting should be stable.  
      -w, --bandwidth                   Select whether to compute bandwidth of reordered matrix.  
      -t[threads], --threads=[threads]  Select the number of threads to run for CPU_BATCH.  
      --f=[perffile],
      --perffile=[perffile]             File to store the performance data. 
      -o[output], --output=[output]     File to store the reorder matrix.   
```

### Sample Usage
To download a few small sample matrices form the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) run  
```
python get_sample_matrices.py
```  
from the `mats/` directory. To reorder one of them change to the `build/` directory and call  
```
./CuthillMcKee ../mats/<some_sample_matrix>.mtx -i ALL -t 24
```  
to run all included Cuthill-McKee implementations. Parallel approaches are run with 1 to 24 threads.  
