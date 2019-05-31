This ZIP contains a source code (spnet2.c) that we used for our study
in our paper "Reproducing Infra-Slow Oscillations with Dopaminergic
Modulation".

*** spnet2.c ***
This code is written in C language.
A typical command to compile the program on a linux/unix computer:

cc spnet2.c -o spnet

You can output raster plots by using this code. For example

./spnet -tau 1 > output.txt

The data size of raster plots are very large (600MB in above example),
and therefore if you are having trouble plotting the data, try
reducing the number of the points to output.

You can obtain firing rates by calculating N_firings/N.

The values of parameters are supplied on the command line,
for example, if you want to let tau = 1, add '-tau 1' to execution command.

Note MT.h was previously available from
http://www.sat.t.u-tokyo.ac.jp/~omi/code/MT.h

