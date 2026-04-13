# ZDA_Adventure
A tool package used to read and convert ZDA files from the Old Rig in Jackson Lab in the Department of Neuroscience of UW-Madison.
The goal of making this package is to let .zda files be directly read and analyzed by Python without using PhotoZ.
With the help of ZDA_Adventure, we can exploit the various packages and units in Pyhthon to do calculations on the Data from the Old Rig.

Currently it contains four basic units: "utility", "tools", "maps" and "identify".
"utility" is used to load the data from .zda files and fix the error in the data array.
"tools" can: apply 3-Degrees polynomial regression to the Data to calculate the Baseline; process the Data by Temporal-Filter and Spatial-Filter.
"maps" will check the Data in several different insights like Maximum Amplitude Map, Signal-to-Noise Map, etc.
"identify" aims to identify the single neuron from SNR (Signal-to-Noise Ratio) Map.

Since it was written in Python language, it will bring some trivial error (if it can be called "error") because PhotoZ was written in C++ and the accuracy of different programming languages is different.
However, I still tried my best to let the output looks like the output from PhotoZ as much as possible.

With the develpment of our research, more functions will be added to this package.
And a using manual will be created soon.
