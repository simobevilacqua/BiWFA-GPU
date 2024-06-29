# A High-performance GPU Implementation of the bidirectional-WFA algorithm

##### Brief description of project:
DNA analysis is a paramount step in personalized medicine, enabling quicker diagnoses and the development of custom drugs.
However, this procedure is extremely compute-intensive and time-consuming. In this context, CPUs cannot process such a large amount of data in a reasonable time.
Therefore, shifting toward hardware-based solutions is key to unlocking the full potential of DNA analysis. In this scenario, this project aims to combine the bidirectional-WFA
algorithm's efficiency and the computational power of GPUs to develop a high-performance solution for sequence alignment.

### Instructions to build and test project
#### Step 1:
Set the correct value for wf_length in headers/biWFA.h

#### Step 2:
Compile the program biWFA.cpp with ```hipcc biWFA.cpp -o biWFA```

#### Step 3:
Run the executable passing the name of the file containing the sequences to align. Use the option -s to show the sequences and their optimal alignment score ```./biWFA sequences/sequences_xxx_x.txt [-s]```

To run the version that allocates the wavefronts in shared memory modify headers/biWFA_shared.h and compile biWFA_shared.cpp
