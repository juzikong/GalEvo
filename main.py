import galaxy_evol
import numpy as np
import plot_simulation_results
from scipy.integrate import quad
from IMFs import Kroupa_IMF
import matplotlib.pyplot as plt
import math
import sys
sys.path.insert(0, 'Generated_IGIMFs')


def function_xi_Kroupa_IMF(mass):
    # integrate this function's output xi result in the number of stars in mass limits.
    xi = Kroupa_IMF.custom_imf(mass, 0)
    return xi

def function_mass_Kroupa_IMF(mass):
    # integrate this function's output m result in the total stellar mass for stars in mass limits.
    m = mass * Kroupa_IMF.custom_imf(mass, 0)
    return m

def simulate(SFR, SFEN, SFE, IMF, SET, check_igimf):
    global distribution
    i = 0
    while i < len(SFR):
        if distribution == "skewnorm":
            generate_sfh_skewnorm(SFR[i], SFEN[i])
        elif distribution == "flat":
            generate_sfh_flat(SFR[i], SFEN[i])
        galaxy_evol.galaxy_evol(imf=IMF, unit_SFR=1, SFE=SFE[i], SFEN=SFEN[i], Z_0=0.00000000134, Z_solar=0.0134,
                                str_evo_table=SET, IMF_name='Kroupa', steller_mass_upper_bound=150,
                                time_resolution_in_Myr=1, mass_boundary_observe_low=3, mass_boundary_observe_up=8,
                                SNIa_ON=True, high_time_resolution=None, plot_show=None, plot_save=None, outflow=None, check_igimf=check_igimf)
        (i) = (i+1)
    return


def generate_sfh_flat(SFR, SFEN):
    # Flat distribution for star formation history
    # took input: star formation rate, star formation event number
    file = open('SFH.txt', 'w')
    j = 0
    while j < SFEN:
        file.write("{}\n".format(10 ** SFR))
        (j) = (j + 1)
    j = 0
    while j < 1000 - SFEN:
        file.write("0\n")
        (j) = (j + 1)
    file.write("# The value in each line stand for the SFR [solar mass / yr]\n")
    file.write("# SFR = the value * unit_SFR (given in file galaxy_evol.py)\n")
    file.write("# in a star formation epoch (10 Myr)\n")
    file.write("# start from time 0 for the first line.\n")
    file.write("# Warning! Effective line number must be larger than 1.\n")
    file.write("# Add a '0' in the next line if there is only one line.\n")

    file.close()
    return


def cal_tot_sf(SFR, SFEN):
    # Skew normal distribution for star formation history
    # took input: maximum star formation rate, star formation event number
    import numpy as np
    from scipy.stats import skewnorm
    # from scipy.stats import f
    global skewness, location
    x = np.linspace(skewnorm.ppf(0.01, skewness, location, 1), skewnorm.ppf(0.99, skewness, location, 1), SFEN)
    y = skewnorm.pdf(x, skewness, location, 1)
    # skewnorm.pdf(x, a, loc, scale) is the location and scale parameters,
    #   [identically equivalent to skewnorm.pdf(y, a) / scale with y = (x - loc) / scale]
    # The scale is not used as the SFEN & SFR setup the scale through parameter tot_sf_set & mult.
    mult = 10 ** SFR / max(y)
    j = 0
    tot_sf = 0
    while j < SFEN:
        sf = mult * y[j]
        tot_sf += sf
        (j) = (j + 1)
    return tot_sf, mult, y


def generate_sfh_skewnorm(SFR, SFEN):
    global sfr_tail
    tot_sf_set = 10 ** SFR * SFEN
    tot_sf = 0
    while tot_sf < tot_sf_set:
        SFEN += 1
        result_cal_tot_sf = cal_tot_sf(SFR, SFEN)
        (tot_sf) = (result_cal_tot_sf[0])

    file = open('SFH.txt', 'w')
    sfr_for_this_epoch = 0
    result_starburst_sf = 0
    result_tail_sf = 0
    j = 0
    while j < SFEN:
        sfr_for_this_epoch = result_cal_tot_sf[1] * result_cal_tot_sf[2][j]
        file.write("{}\n".format(sfr_for_this_epoch))
        if sfr_for_this_epoch > 10 ** SFR/2:
            result_starburst_sf += sfr_for_this_epoch
        else:
            result_tail_sf += sfr_for_this_epoch
        (j) = (j + 1)
    sfr_for_the_tail_epoch = sfr_for_this_epoch / 2
    if sfr_tail == 0:
        j = 0
        while j < 1000 - SFEN:
            file.write("0\n")
            (j) = (j + 1)
    elif sfr_tail == 1:
        j = 0
        while j < 100:
            file.write("{}\n".format(sfr_for_the_tail_epoch))
            result_tail_sf += sfr_for_the_tail_epoch
            (j) = (j + 1)
        while j < 1000 - SFEN:
            file.write("0\n")
            (j) = (j + 1)
    file.write("# The value in each line stand for the SFR [solar mass / yr]\n")
    file.write("# SFR = the value * unit_SFR (given in file galaxy_evol.py)\n")
    file.write("# in a star formation epoch (10 Myr)\n")
    file.write("# start from time 0 for the first line.\n")
    file.write("# Warning! Effective line number must be larger than 1.\n")
    file.write("# Add a '0' in the next line if there is only one line.\n")

    if sfr_tail == 1:
        print("star formation tail (after the SFR is lower than half of the maximum value) contributes {}% "
              "of the total star formation.".format(round(result_tail_sf/(result_starburst_sf+result_tail_sf)*100, 2)))

    file.close()
    return

def F05(mass_range_1, mass_range_2):

    file = open('simulation_results/imf:{}-SFE:{}-log_SFR:{}-SFEN:{}.txt'.format(IMF, SFE[0], SFR[0], SFEN[0]), 'r')
    number_of_sf_epoch = int(file.readlines()[1])
    file.close()

    i = 0
    igimfs = []
    while i < number_of_sf_epoch:
        igimf_file_name = "igimf_epoch_{}".format(i)
        igimfs.append(__import__(igimf_file_name))
        (i) = (i+1)

    integrate_Kroupa_stellar_mass_range_1 = quad(function_mass_Kroupa_IMF, mass_range_1[0], mass_range_1[1], limit=50)[
        0]
    integrate_Kroupa_stellar_mass_range_2 = quad(function_mass_Kroupa_IMF, mass_range_2[0], mass_range_2[1], limit=50)[
        0]
    integrate_Kroupa_stellar_number_mass_range_1 = \
    quad(function_xi_Kroupa_IMF, mass_range_1[0], mass_range_1[1], limit=50)[0]
    integrate_Kroupa_stellar_number_mass_range_2 = \
    quad(function_xi_Kroupa_IMF, mass_range_2[0], mass_range_2[1], limit=50)[0]

    F_mass_Kroupa_IMF = integrate_Kroupa_stellar_mass_range_1 / integrate_Kroupa_stellar_mass_range_2
    F_number_Kroupa_IMF = integrate_Kroupa_stellar_number_mass_range_1 / integrate_Kroupa_stellar_number_mass_range_2

    integrate_IGIMF_stellar_mass_range_1 = 0
    integrate_IGIMF_stellar_mass_range_2 = 0
    integrate_IGIMF_stellar_number_mass_range_1 = 0
    integrate_IGIMF_stellar_number_mass_range_2 = 0
    i = 0
    while i < number_of_sf_epoch:
        def function_xi_IGIMF(mass):
            xi = igimfs[i].custom_imf(mass, 0)
            return xi

        def function_mass_IGIMF(mass):
            m = mass * igimfs[i].custom_imf(mass, 0)
            return m

        integrate_IGIMF_stellar_mass_range_1 += quad(function_mass_IGIMF, mass_range_1[0], mass_range_1[1], limit=50)[0]
        integrate_IGIMF_stellar_mass_range_2 += quad(function_mass_IGIMF, mass_range_2[0], mass_range_2[1], limit=50)[0]
        integrate_IGIMF_stellar_number_mass_range_1 += \
        quad(function_xi_IGIMF, mass_range_1[0], mass_range_1[1], limit=50)[0]
        integrate_IGIMF_stellar_number_mass_range_2 += \
        quad(function_xi_IGIMF, mass_range_2[0], mass_range_2[1], limit=50)[0]
        (i) = (i + 1)

    F_mass_IGIMF = integrate_IGIMF_stellar_mass_range_1 / integrate_IGIMF_stellar_mass_range_2
    F_number_IGIMF = integrate_IGIMF_stellar_number_mass_range_1 / integrate_IGIMF_stellar_number_mass_range_2

    print("F_mass_Kroupa_IMF", F_mass_Kroupa_IMF)
    print("F_mass_IGIMF", F_mass_IGIMF)
    print("F_number_Kroupa_IMF", F_number_Kroupa_IMF)
    print("F_number_IGIMF", F_number_IGIMF)
    return


if __name__ == '__main__':

    #########################
    IMF = "igimf"
    distribution = "flat"  # Change the shape of the SFH. Can be "flat" or "skewnorm"
    SET = "portinari98"
    skewness = 0  # the alpha parameter in standard Skew normal distribution
                  # the normal distribution is recovered when skewness = 0
    sfr_tail = 0  # It is possible to set a long tail of low SFR SFH.
                  # But you have to look into the sfr_tail settings in the code above.
    location = 0  # Shift the skew normal destribution.

    #######################################################
    # # To study the F05 for given SFH and F05 definition:

    mass_range_1 = [0.3, 0.4]
    mass_range_2 = [0.08, 1]
    IMF = "igimf"
    distribution = "skewnorm"
    skewness = 10
    SFR = [1.0]
    SFEN = [3]
    SFE = [0.9]
    # simulate(SFR, SFEN, SFE, IMF, SET, check_igimf=False)  # You can trun of simulation at the seconed run
                                                           # if you did't change the SFR, SFEN, and SFE.
    # plot the generated SFH
    SFH_input = np.loadtxt('SFH.txt')
    i = 0
    while i < len(SFH_input):
        if SFH_input[i] == 0:
            length_SFH = i
            break
        (i) = (i+1)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(0, figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1)
    SFR_list = []
    age_list = []
    age_list.append(0)
    SFR_list.append(-5)
    for i in range(length_SFH):
        age_list.append(i * 10)
        SFR_list.append(math.log(SFH_input[i], 10))
        age_list.append((i + 1) * 10)
        SFR_list.append(math.log(SFH_input[i], 10))
    age_list.append((i + 1) * 10)
    SFR_list.append(-5)
    plt.plot(age_list, SFR_list)
    plt.xlabel(r'time [10 Myr]')
    plt.ylabel('log$_{10}$(SFR) [solar mass/yr]')
    plt.tight_layout()
    plt.show()


    #######################################################
    # #  To study the galaxy wide metal yield for different galaxy masses and the "downsizing" relation:

    # # # igimf best fit:
    # IMF = "igimf"
    # distribution = "flat"
    # SFR = [-2.0, -1.0, 0.0, 1.0, 2.0]
    # SFEN = [30, 35, 40, 50, 100]
    # SFE = [0.3, 0.7, 0.9, 0.9, 0.9]
    # simulate(SFR, SFEN, SFE, IMF, SET, check_igimf=True)  # You can trun of simulation at the seconed run
    #                                     # if you did't change the SFR, SFEN, and SFE.


    if len(SFR) == 1:
        F05(mass_range_1, mass_range_2)
    elif len(SFR) > 1:
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        fig = plt.figure(1, figsize=(6, 5.25))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(SFR, SFD)
        plt.xlabel(r'log$_{10}$(SFR [M$_\odot$/yr])')
        plt.ylabel('$\delta$t [Gyr]')
        plt.tight_layout()
        plt.show()
        plot_simulation_results.plot(plot_star=1, plot_gas=0, plot_mid=0, x_axis="LumDyn", SFE=SFE, SFEN=SFEN,
                                     log_SFR=SFR, IMF=IMF)
        # x_axis = "dynamical" or "luminous" or "SEDfit"
        # a final dynamical mass being 8.5 -- 11.5 lies in the observed velocity dispersion range.
