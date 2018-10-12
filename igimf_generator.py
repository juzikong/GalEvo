def function_generate_igimf_file(SFR=None, Z_over_H=None, printout=False, sf_epoch=0, check=False):
    # python3 code, last update Wed 10 July 2017


    # --------------------------------------------------------------------------------------------------------------------------------
    # import modules and libraries
    # --------------------------------------------------------------------------------------------------------------------------------

    import galIMF  # galIMF containing IGIMF function and OSGIMF function and additional computational modules

    import matplotlib.pyplot as plt  # matplotlib for plotting
    import numpy as np
    from scipy.integrate import simps  # numpy and scipi for array operations
    import math
    import time
    import sys


    # --------------------------------------------------------------------------------------------------------------------------------
    # check if the required IGIMF has already been generated
    # --------------------------------------------------------------------------------------------------------------------------------


    check_file = open('Generated_IGIMFs/all_igimf_list.txt', 'r')
    igimf_list = check_file.readlines()
    check_file.close()

    exist = 0

    if check == True:
        i = 0
        while i < len(igimf_list):
            data = [float(x) for x in igimf_list[i].split()]
            if SFR == data[0] and Z_over_H == data[1]:
                exist = 1
                break
            (i) = (i + 1)

    if exist == 0:

        # --------------------------------------------------------------------------------------------------------------------------------
        # add new headline into the list file -- all_igimf_list.txt:
        # --------------------------------------------------------------------------------------------------------------------------------

        check_file = open('Generated_IGIMFs/all_igimf_list.txt', 'r')
        igimf_list = check_file.read()
        check_file.close()

        check_file = open('Generated_IGIMFs/all_igimf_list.txt', 'w')
        new_headline = igimf_list + '{} {}\n'.format(SFR, Z_over_H)
        check_file.write(new_headline)
        check_file.close()

        # --------------------------------------------------------------------------------------------------------------------------------
        # Define code parameters necesarry for the computations:
        # --------------------------------------------------------------------------------------------------------------------------------

        # the most crutial ones, which you most likely might want to change

        if SFR == None:
            SFR = float(
                input(
                    "Please input the galaxy-wide SFR in solar mass per year and ended the input with the return key. "
                    "(A typical input SFR is from 0.0001 to 10000. "
                    "We recommed a value smallar than 0.01 for the first run as high SFR calculations take more time.)\n"
                    "You can input 1e-4 as 0.0001\n"
                    "\nSFR [Msolar/yr] = "))
            # Star Formation Rate [solar mass / yr]
        if SFR != 0:
            bindw = galIMF.resolution_histogram_relative = 10 ** (max((0 - math.log(SFR, 10)), 0) ** (0.2) - 1.9)
        # will change the resolution of histogram for optimall sampling automatically addjusted with SFR value.
        alpha3_model = 1  # IMF high-mass-end power-index model, see Function_alpha_3_change in file 'galIMF.py'
        alpha_2 = 2.3  # IMF middle-mass power-index
        alpha_1 = 1.3  # IMF low-mass-end power-index
        alpha2_model = 1  # see file 'galIMF.py'
        alpha1_model = 1  # see file 'galIMF.py'
        beta_model = 1
        if Z_over_H == None:
            Z_over_H = float(input("\nPlease input the metallicity, [Z/H] = log(M_{metal}/M_{H})-log(M_{metal,sun}/M_{H,sun})"
                                    "\n\n[Z/H] = ..."))
        # ----------------------------------------------------------------

        # Parameters below are internal parameters of the theory.
        # Read Yan et al. 2017 carefully before change them!

        delta_t = 10.  # star formation epoch [Myr]
        I_ecl = 1.  # normalization factor in the Optimal Sampling condition equation
        M_ecl_U = 10**9 # 10**(0.75 * math.log(SFR, 10) + 4.8269) # Recchi 2009
        # 10 ** 15  # embedded cluster mass upper limit [solar mass]
        M_ecl_L = 5.  # embedded cluster mass lower limit [solar mass]
        I_str = 1.  # normalization factor in the Optimal Sampling condition equation
        M_str_L = 0.08  # star mass lower limit [solar mass]
        M_turn = 0.5  # IMF power-index breaking mass [solar mass]
        M_turn2 = 1.  # IMF power-index breaking mass [solar mass]
        M_str_U = 150  # star mass upper limit [solar mass]

        if printout == True:
            print("\n - GalIMF run in progress..")
        start_time = time.time()

        # --------------------------------------------------------------------------------------------------------------------------------
        # Construct IGIMF:
        # --------------------------------------------------------------------------------------------------------------------------------

        if printout == True:
            print("\nCalculating IGIMF......")

        galIMF.function_galIMF(
            "I",  # IorS ### "I" for IGIMF; "OS" for OSGIMF
            SFR,  # Star Formation Rate [solar mass / yr]
            alpha3_model,  # IMF high-mass-end power-index model, see file 'alpha3.py'
            delta_t,  # star formation epoch [Myr]
            Z_over_H,
            I_ecl,  # normalization factor in the Optimal Sampling condition equation
            M_ecl_U,  # embedded cluster mass upper limit [solar mass]
            M_ecl_L,  # embedded cluster mass lower limit [solar mass]
            beta_model,  ### ECMF power-index model, see file 'beta.py'
            I_str,  # normalization factor in the Optimal Sampling condition equation
            M_str_L,  # star mass lower limit [solar mass]
            alpha_1,  # IMF low-mass-end power-index
            alpha1_model,  # see file 'alpha1.py'
            M_turn,  # IMF power-index change point [solar mass]
            alpha_2,  # IMF middle-mass power-index
            alpha2_model,  # see file 'alpha2.py'
            M_turn2,  # IMF power-index change point [solar mass]
            M_str_U,  # star mass upper limit [solar mass]
            printout
        )

        if printout == True:
            ### Decorate the output file ###
            igimf_raw = np.loadtxt('GalIMF_IGIMF.txt')
            if M_str_U - igimf_raw[-1][0] > 0.01:
                file = open('GalIMF_IGIMF.txt', 'a')
                file.write("{} 0\n\n".format(igimf_raw[-1][0] + 0.01))
                file.write("{} 0".format(M_str_U))
                file.close()
            else:
                file = open('GalIMF_IGIMF.txt', 'a')
                file.write("{} 0".format(M_str_U))
                file.close()

        global masses, igimf

        masses = np.array(galIMF.List_M_str_for_xi_str)
        igimf = np.array(galIMF.List_xi)


        #######################################################
        # generated igimf is normalized by default to a total mass formed in 10 Myr given the SFR,
        # i.e., total stellar mass!!!
        # to change the normalization follow the commented part:
        #######################################################
        # Norm = simps(igimf*masses,masses) #- normalization to a total mass
        # Norm = simps(igimf,masses) #- normalization to number of stars
        # Mtot1Myr = SFR*10*1.e6 #total mass formed in 10 Myr
        # igimf = np.array(igimf)*Mtot1Myr/Norm
        #######################################################


        global length_of_igimf
        length_of_igimf = len(igimf)

        def write_imf_input2():
            print("Generating new igimf.")
            global file, masses, igimf
            if SFR == 0:
                file = open('Generated_IGIMFs/igimf_SFR_Zero.py', 'w')
                file.write("def custom_imf(mass, time):  # there is no time dependence for IGIMF\n")
                file.write("    return 0\n")
                file.close()
            else:
                file = open('Generated_IGIMFs/igimf_SFR_{}_Fe_over_H_{}.py'.format(round(math.log(SFR, 10)*100000),
                                                                                   round(Z_over_H * 100000)), 'w')
                file.write("# File to define a custom IMF\n"
                           "# The return value represents the chosen IMF value for the input mass\n\n\n")
                file.write("def custom_imf(mass, time):  # there is no time dependence for IGIMF\n")
                file.write("    if mass < 0.08:\n")
                file.write("        return 0\n")
                file.write("    elif mass < %s:\n" % masses[1])
                k = (igimf[0] - igimf[1]) / (masses[0] - masses[1])
                b = igimf[0] - k * masses[0]
                file.write("        return {} * mass + {}\n".format(k, b))
                write_imf_input_middle2(1)
                file.write("    else:\n")
                file.write("        return 0\n")
                file.close()
            return

        def write_imf_input_middle2(i):
            global file, length_of_igimf
            while i < length_of_igimf - 1:
                file.write("    elif mass < %s:\n" % masses[i + 1])
                k = (igimf[i] - igimf[i + 1]) / (masses[i] - masses[i + 1])
                b = igimf[i] - k * masses[i]
                file.write("        return {} * mass + {}\n".format(k, b))
                (i) = (i + 3)
            return

        write_imf_input2()

        def write_imf_input3():
            print("Generating new igimf.")
            global file, masses, igimf
            if SFR == 0:
                file = open('Generated_IGIMFs/igimf_epoch_{}.py'.format(sf_epoch), 'w')
                file.write("def custom_imf(mass, time):  # there is no time dependence for IGIMF\n")
                file.write("    return 0\n")
                file.close()
            else:
                file = open('Generated_IGIMFs/igimf_epoch_{}.py'.format(sf_epoch), 'w')
                file.write("# File to define a custom IMF\n"
                           "# The return value represents the chosen IMF value for the input mass\n\n\n")
                file.write("def custom_imf(mass, time):  # there is no time dependence for IGIMF\n")
                file.write("    if mass < 0.08:\n")
                file.write("        return 0\n")
                file.write("    elif mass < %s:\n" % masses[1])
                k = (igimf[0] - igimf[1]) / (masses[0] - masses[1])
                b = igimf[0] - k * masses[0]
                file.write("        return {} * mass + {}\n".format(k, b))
                write_imf_input_middle2(1)
                file.write("    else:\n")
                file.write("        return 0\n")
                file.close()
            return

        write_imf_input3()

        if printout == True:
            print("imf_input.py rewritten for SFR = {} and metallicity = {}\n".format(SFR, Z_over_H))

            file = open('../gimf_Fe_over_H.txt', 'w')
            file.write("{}".format(Z_over_H))
            file.close()

            file = open('../gimf_SFR.txt', 'w')
            file.write("{}".format(SFR))
            file.close()

            print(" - GalIMF run completed - Run time: %ss -\n\n" % round((time.time() - start_time), 2))
        return
    return

if __name__ == '__main__':
    function_generate_igimf_file()