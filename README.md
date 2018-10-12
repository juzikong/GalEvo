# GalEvo trial version

What to do:

1. Run visualize_stellar_yield_table.py to check the stellar metal yield. 
This applied yield can not be changed but I will provide more options in the future.

2. You can set the star formation history by rewriting the SFH.txt file. Have a look at this file.

3. Run galaxy_evol.py will simulate the galaxy evolution with given SFH (as in SFH.txt) for 10 Gyr. The output are shown in the generated plots.

4. Run main.py. But first read the bottom part of it, i.e., after "if __name__ == '__main__':". 
It contains two mode: 
(a) the F05 mode requires a single SFH input.
(b) the chemial-yield--galaxy-mass mode requires a list of SFH input.

Better use main.py instead of galaxy_evol.py when you can. 
As it will check whether the simulation under the same initialization has been performed and save a lot of time.
