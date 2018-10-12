import matplotlib.pyplot as plt
import math
import os.path


def plot(plot_star=1, plot_gas=1, plot_mid=0, x_axis="dynamical", SFE=[0.5, 0.5], SFEN=[], log_SFR=[7, 8], IMF="Kroupa"):

    # SFR = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]  # log_10 star formation rate [solar mass/yr]
    # SFD = [2.65, 1.77, 1.24, 0.93, 0.67, 0.47, 0.32, 0.22, 0.15, 0.1]  # star formation duration [Gyr]
    # SFD2 = [2.75, 1.87, 1.34, 1.03, 0.77, 0.57, 0.42, 0.32, 0.25, 0.2]  # star formation duration [Gyr]
    #
    # plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='x-small')
    # plt.rc('ytick', labelsize='x-small')
    # fig = plt.figure(0, figsize=(6, 5.25))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.plot(SFR, SFD)
    # plt.plot(SFR, SFD2)
    # plt.xlabel(r'log$_{10}$(SFR [M$_\odot$/yr])')
    # plt.ylabel('$\delta$t [Gyr]')
    # # plt.xlim(6.4, 1.01 * log_time_axis[-1])
    # # plt.ylim(-3, 0.5)


    ########################

    final_stellar_mass_list = []  # x-axis
    final_stellar_mass_list_dif = []  # x-axis
    LumDyn_list = []  # x-axis
    final_dynamical_mass_list = []  # x-axis
    final_dynamical_mass_list_dif = []  # x-axis
    total_mas_in_box_list = []  # x-axis

    # plot_at_age = [1 * 10 ** 8, 1 * 10 ** 9, 9 * 10 ** 9, 10 * 10 ** 9, 11 * 10 ** 9]
    plot_at_age = [5 * 10 ** 7, 1 * 10 ** 8, 5 * 10 ** 8, 1 * 10 ** 9, 1 * 10 ** 10]
    length_plot_at_age = len(plot_at_age)

    # Dabringhausen 2008 eq.4
    Dabringhausen_2008_a = 2.95
    Dabringhausen_2008_b = 0.596
    gravitational_binding_energy_list = []

    Number_of_SN_at_all_time_step_list = [[]]
    Gas_Fe_over_H_at_all_step_list = [[]]
    Star_Fe_over_H_at_all_step_list = [[]]
    Gas_Mg_over_H_at_all_step_list = [[]]
    Star_Mg_over_H_at_all_step_list = [[]]
    Gas_O_over_H_at_all_step_list = [[]]
    Star_O_over_H_at_all_step_list = [[]]
    Gas_Mg_over_Fe_at_all_step_list = [[]]
    Star_Mg_over_Fe_at_all_step_list = [[]]
    Gas_O_over_Fe_at_all_step_list = [[]]
    Star_O_over_Fe_at_all_step_list = [[]]
    Gas_metallicity_at_all_step_list = [[]]
    Star_metallicity_at_all_step_list = [[]]
    i = 0
    while i < length_plot_at_age - 1:
        Number_of_SN_at_all_time_step_list.append([])
        Gas_Fe_over_H_at_all_step_list.append([])
        Star_Fe_over_H_at_all_step_list.append([])
        Gas_Mg_over_H_at_all_step_list.append([])
        Star_Mg_over_H_at_all_step_list.append([])
        Gas_O_over_H_at_all_step_list.append([])
        Star_O_over_H_at_all_step_list.append([])
        Gas_Mg_over_Fe_at_all_step_list.append([])
        Star_Mg_over_Fe_at_all_step_list.append([])
        Gas_O_over_Fe_at_all_step_list.append([])
        Star_O_over_Fe_at_all_step_list.append([])
        Gas_metallicity_at_all_step_list.append([])
        Star_metallicity_at_all_step_list.append([])
        (i) = (i + 1)

    ##########################

    raw_data = []
    i = 0
    while i < len(log_SFR):
        if SFEN == []:
            file = open(
                'simulation_results/imf:{}-SFE:{}-log_SFR:{}.txt'.format(IMF, SFE[i], log_SFR[i]),'r')
        else:
            file = open('simulation_results/imf:{}-SFE:{}-log_SFR:{}-SFEN:{}.txt'.format(IMF, SFE[i], log_SFR[i], SFEN[i]), 'r')
        # file = open('simulation_results_0.1Gyr/imf:igimf-SFE:{}-log_SFR:{}.txt'.format(SFE[i], log_SFR[i]), 'r')
        raw_data.append(file.readlines())
        file.close()
        (i) = (i+1)

    k = 0
    while k < len(raw_data):
        total_mas_in_box = float(raw_data[k][13])
        total_mas_in_box_list.append(math.log(total_mas_in_box, 10))
        final_stellar_mass = float(raw_data[k][7])
        final_stellar_mass_list.append(final_stellar_mass)
        final_stellar_mass_list_dif.append(10**final_stellar_mass/total_mas_in_box)
        final_dynamical_mass = float(raw_data[k][11])
        final_dynamical_mass_list.append(final_dynamical_mass)
        final_dynamical_mass_list_dif.append(10**final_dynamical_mass/total_mas_in_box)
        log_sigma = math.log(0.86, 10) + 0.22 * final_dynamical_mass
        LumDyn_list.append(log_sigma)



        expansion_factor = 5  # the expansion_factor should be a function of galaxy final_stellar_mass
        # and rise with the mass
        # See Kroupa 2008 for instantaneous and adibatic expansion
        log_binding_energy = round(
            math.log(4.3 * 6 / 5, 10) + 40 + (2 - Dabringhausen_2008_b) * math.log(total_mas_in_box, 10)
            - math.log(Dabringhausen_2008_a, 10) + 6 * Dabringhausen_2008_b +
            math.log(expansion_factor, 10), 1)
        gravitational_binding_energy_list.append(log_binding_energy)

        time = [float(x) for x in raw_data[k][15].split()]
        Number_of_SN = [math.log(float(x), 10) for x in raw_data[k][17].split()]
        Gas_Fe_over_H = [float(x) for x in raw_data[k][19].split()]
        Star_Fe_over_H = [float(x) for x in raw_data[k][21].split()]
        Gas_Mg_over_Fe = [float(x) for x in raw_data[k][23].split()]
        Star_Mg_over_Fe = [float(x) for x in raw_data[k][25].split()]
        Gas_O_over_Fe = [float(x) for x in raw_data[k][27].split()]
        Star_O_over_Fe = [float(x) for x in raw_data[k][29].split()]
        Gas_Mg_over_H = [float(x) for x in raw_data[k][31].split()]
        Star_Mg_over_H = [float(x) for x in raw_data[k][33].split()]
        Gas_O_over_H = [float(x) for x in raw_data[k][35].split()]
        Star_O_over_H = [float(x) for x in raw_data[k][37].split()]
        Gas_metallicity = [float(x) for x in raw_data[k][39].split()]
        Star_metallicity = [float(x) for x in raw_data[k][41].split()]

        plot_at_age_time_index = []
        j = 0
        while j < length_plot_at_age:
            i = 0
            while i < len(time):
                if time[i] == plot_at_age[j]:
                    plot_at_age_time_index.append(i)
                (i) = (i + 1)
            (j) = (j + 1)

        i = 0
        while i < length_plot_at_age:
            Number_of_SN_at_all_time_step_list[i].append([Number_of_SN[plot_at_age_time_index[i]] + 52])
            Gas_Fe_over_H_at_all_step_list[i].append(Gas_Fe_over_H[plot_at_age_time_index[i]])
            Star_Fe_over_H_at_all_step_list[i].append(Star_Fe_over_H[plot_at_age_time_index[i]])
            Gas_Mg_over_H_at_all_step_list[i].append(Gas_Mg_over_H[plot_at_age_time_index[i]])
            Star_Mg_over_H_at_all_step_list[i].append(Star_Mg_over_H[plot_at_age_time_index[i]])
            Gas_O_over_H_at_all_step_list[i].append(Gas_O_over_H[plot_at_age_time_index[i]])
            Star_O_over_H_at_all_step_list[i].append(Star_O_over_H[plot_at_age_time_index[i]])
            Gas_Mg_over_Fe_at_all_step_list[i].append(Gas_Mg_over_Fe[plot_at_age_time_index[i]])
            Star_Mg_over_Fe_at_all_step_list[i].append(Star_Mg_over_Fe[plot_at_age_time_index[i]])
            Gas_O_over_Fe_at_all_step_list[i].append(Gas_O_over_Fe[plot_at_age_time_index[i]])
            Star_O_over_Fe_at_all_step_list[i].append(Star_O_over_Fe[plot_at_age_time_index[i]])
            Gas_metallicity_at_all_step_list[i].append(Gas_metallicity[plot_at_age_time_index[i]])
            Star_metallicity_at_all_step_list[i].append(Star_metallicity[plot_at_age_time_index[i]])
            (i) = (i + 1)
        (k) = (k+1)



    ##########################
    #         Plot           #
    ##########################


    if x_axis == "dynamical":
        mass_list = final_dynamical_mass_list
    elif x_axis == "luminous":
        mass_list = final_stellar_mass_list
    elif x_axis == "LumDyn":
        mass_list = LumDyn_list
    elif x_axis == "SFR":
        mass_list = log_SFR

    # #################################
    #
    # plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='x-small')
    # plt.rc('ytick', labelsize='x-small')
    # fig = plt.figure(0, figsize=(6, 5.25))
    # ax = fig.add_subplot(1, 1, 1)
    #
    # plt.plot([log_SFR[0], log_SFR[-1]], [total_mas_in_box_list[0], total_mas_in_box_list[-1]], ls="dotted", c="k", label=r'M$_{galaxy}$=M$_{baryon}$')
    # plt.plot([log_SFR[0], log_SFR[-1]], [total_mas_in_box_list[0]-0.301, total_mas_in_box_list[-1]-0.301], ls="dashed", c="k", label=r'M$_{galaxy,ini}$')
    # plt.plot(log_SFR, final_dynamical_mass_list, label='dynamical (alive+remnant)')
    # plt.plot(log_SFR, final_stellar_mass_list, label='luminous (alive)')
    #
    # plt.xlabel(r'log$_{10}$(gwSFR) [M$_\odot$/yr]')
    # # plt.xlabel(r'log$_{10}$(M$_{baryon}$) [M$_\odot$]')
    # plt.ylabel(r'log$_{10}$(M$_{galaxy}$) [M$_\odot$]')
    #
    # plt.tight_layout()
    # plt.legend(prop={'size': 10}, loc='best')
    #
    # #################################
    #
    # plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='x-small')
    # plt.rc('ytick', labelsize='x-small')
    # fig = plt.figure(-1, figsize=(6, 5.25))
    # ax = fig.add_subplot(1, 1, 1)
    #
    # plt.plot([log_SFR[0], log_SFR[-1]], [1, 1], ls="dashed", c="k", label=r'M$_{galaxy}$=M$_{baryon}$')
    # plt.plot([log_SFR[0], log_SFR[-1]], [0.5, 0.5], ls="dotted", c="k", label=r'M$_{galaxy,ini}$')
    # plt.plot(log_SFR, final_dynamical_mass_list_dif, label='dynamical (alive+remnant)')
    # plt.plot(log_SFR, final_stellar_mass_list_dif, label='luminous (alive)')
    #
    # plt.xlabel(r'log$_{10}$(gwSFR) [M$_\odot$/yr]')
    # # plt.xlabel(r'log$_{10}$(M$_{baryon}$) [M$_\odot$]')
    # plt.ylabel(r'M$_{galaxy}$/M$_{baryon}$')
    #
    # plt.tight_layout()
    # plt.legend(prop={'size': 10}, loc='best')


    # ##########################
    # #      Plot energy       #
    # ##########################
    #
    # plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='x-small')
    # plt.rc('ytick', labelsize='x-small')
    # fig = plt.figure(1, figsize=(6, 5.25))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.plot(mass_list, gravitational_binding_energy_list, ls='dashed', c='k',
    #          label='Gravitational binding energy')
    # plt.plot([], [], c="k", label='SN energy at different time')
    # plt.plot(mass_list, Number_of_SN_at_all_time_step_list[4], c="brown", label='10 Gyr')
    # plt.plot(mass_list, Number_of_SN_at_all_time_step_list[3], c="red", label='1 Gyr')
    # plt.plot(mass_list, Number_of_SN_at_all_time_step_list[2], c="orange", label='500 Myr')
    # plt.plot(mass_list, Number_of_SN_at_all_time_step_list[1], c="green", label='100 Myr')
    # plt.plot(mass_list, Number_of_SN_at_all_time_step_list[0], c="blue", label='50 Myr')
    #
    # if x_axis == "dynamical":
    #     plt.xlabel(r'log$_{10}$(galaxy dynamical mass) [M$_\odot$]')
    # elif x_axis == "luminous":
    #     plt.xlabel(r'log$_{10}$(galaxy luminous mass) [M$_\odot$]')
    # elif x_axis == "LumDyn":
    #     plt.xlabel(r'log$_{10}$(\sigma) [km/s]')
    # elif x_axis == "SFR":
    #     plt.xlabel(r'log$_{10}$(gwSFR) [M$_\odot$/yr]')
    # plt.ylabel('Energy [erg]')
    # # plt.xlim(6.4, 1.01 * log_time_axis[-1])
    # # plt.ylim(-3, 0.5)
    # plt.tight_layout()
    # plt.legend(prop={'size': 10}, loc='best')

    ##########################
    #        Plot [Z/H]      #
    ##########################

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(2, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot([], [], c="k", ls='dashed', label='gas')
    plt.plot([], [], c="k", label='stellar')
    plt.plot([], [], c="blue", label='50 Myr')
    plt.plot([], [], c="green", label='100 Myr')
    plt.plot([], [], c="orange", label='500 Myr')
    plt.plot([], [], c="red", label='1 Gyr')
    plt.plot([], [], c="brown", label='10 Gyr')

    plt.plot([1, 3], [0, 0], ls='dotted', lw=1, label='Solar Value')

    plt.plot([1.8, 2.4], [0.02, 0.16], ls='dashed', c='k', lw=1, label='Johansson 2012')  # Johansson 2012 421-1908
    plt.plot([1.8, 2.4], [-0.2, 0.2], ls='dashed', c='k', lw=1, label='Thomas 2010')  # Thomas 404, 1775–1789 (2010)
    plt.plot([2.1, 2.5], [0.1, 0.3], ls='dashed', c='k', lw=1, label='Thomas 2005')  # Thomas 621:673–694, 2005
    plt.plot([2.0, 2.4], [-0.168, 0.1984], ls='dashed', c='k', lw=1, label='Graves 2007')  # Graves et al. 671:243-271, 2007
    plt.plot([2.0, 2.4], [0.02, 0.378], ls='dashed', c='k', lw=1, label='Graves 2007')  # Graves et al. 671:243-271, 2007

    if plot_gas == 1:
        plt.plot(mass_list, Gas_metallicity_at_all_step_list[0], c="blue", ls='dashed')
        plt.plot(mass_list, Gas_metallicity_at_all_step_list[1], c="green", ls='dashed')
        plt.plot(mass_list, Gas_metallicity_at_all_step_list[2], c="orange", ls='dashed')
        plt.plot(mass_list, Gas_metallicity_at_all_step_list[3], c="red", ls='dashed')
        plt.plot(mass_list, Gas_metallicity_at_all_step_list[4], c="brown", ls='dashed')
    if plot_star == 1:
        plt.plot(mass_list, Star_metallicity_at_all_step_list[0], c="blue")
        plt.plot(mass_list, Star_metallicity_at_all_step_list[1], c="green")
        plt.plot(mass_list, Star_metallicity_at_all_step_list[2], c="orange")
        plt.plot(mass_list, Star_metallicity_at_all_step_list[3], c="red")
        plt.plot(mass_list, Star_metallicity_at_all_step_list[4], c="brown")
    if plot_mid == 1:
        mid0 = []
        mid1 = []
        mid2 = []
        mid3 = []
        mid4 = []
        i = 0
        while i < len(Gas_metallicity_at_all_step_list[0]):
            mid0.append((Gas_metallicity_at_all_step_list[0][i] + Star_metallicity_at_all_step_list[0][i]) / 2)
            mid1.append((Gas_metallicity_at_all_step_list[1][i] + Star_metallicity_at_all_step_list[1][i]) / 2)
            mid2.append((Gas_metallicity_at_all_step_list[2][i] + Star_metallicity_at_all_step_list[2][i]) / 2)
            mid3.append((Gas_metallicity_at_all_step_list[3][i] + Star_metallicity_at_all_step_list[3][i]) / 2)
            mid4.append((Gas_metallicity_at_all_step_list[4][i] + Star_metallicity_at_all_step_list[4][i]) / 2)
            (i) = (i + 1)
        plt.plot(mass_list, mid0, c="blue", ls='dotted')
        plt.plot(mass_list, mid1, c="green", ls='dotted')
        plt.plot(mass_list, mid2, c="orange", ls='dotted')
        plt.plot(mass_list, mid3, c="red", ls='dotted')
        plt.plot(mass_list, mid4, c="brown", ls='dotted')

    if x_axis == "dynamical":
        plt.xlabel(r'log$_{10}$(galaxy dynamical mass) [M$_\odot$]')
    elif x_axis == "luminous":
        plt.xlabel(r'log$_{10}$(galaxy luminous mass) [M$_\odot$]')
    elif x_axis == "LumDyn":
        plt.xlabel(r'log$_{10}$(\sigma) [km/s]')
    elif x_axis == "SFR":
        plt.xlabel(r'log$_{10}$(gwSFR) [M$_\odot$/yr]')
    plt.ylabel('[Z/H]')
    # plt.xlim(6.4, 1.01 * log_time_axis[-1])
    # plt.ylim(-3, 0.5)
    plt.tight_layout()
    plt.legend(prop={'size': 10}, loc='best')

    ##########################
    ####   Plot [Fe/H]   #####
    ##########################

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(3, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot([], [], c="k", ls='dashed', label='gas')
    plt.plot([], [], c="k", label='stellar')
    plt.plot([], [], c="blue", label='50 Myr')
    plt.plot([], [], c="green", label='100 Myr')
    plt.plot([], [], c="orange", label='500 Myr')
    plt.plot([], [], c="red", label='1 Gyr')
    plt.plot([], [], c="brown", label='10 Gyr')

    plt.plot(mass_list, [0] * len(mass_list), ls='dotted', c='0.5', lw=3, label='Solar Value')

    plt.plot([1.9, 2.5], [-0.07, 0], ls='dashed', c='k', lw=1, label='Conroy 2014')  # Conroy 2014 780-33
    plt.plot([1.9, 2.5], [-0.17, 0.03], ls='dashed', c='k', lw=1, label='Graves 2008')  # Graves & Schiavon 2008
    plt.plot([1.8, 2.6], [-0.07, -0.1], ls='dashed', c='k', lw=1, label='Johansson 2012')  # Johansson 2012
    plt.plot([1, 1.7, 2.5], [-1.5, 0, 0], ls='dashed', c='k', lw=1, label='Koleva 2011')  # Koleva et al. 417, 1643–1671 (2011)
    plt.plot([2, 2.5], [-0.24, 0], ls='dashed', c='k', lw=1, label='Graves 2007')  # Graves et al. 671:243-271, 2007
    plt.plot([1.7, 2.48], [0, 0], ls='dashed', c='k', lw=1, label='Eigenthaler 2013')  # Eigenthaler & Zeilinger2 A&A 553, A99 (2013)

    if plot_gas == 1:
        plt.plot(mass_list, Gas_Fe_over_H_at_all_step_list[0], c="blue", ls='dashed')
        plt.plot(mass_list, Gas_Fe_over_H_at_all_step_list[1], c="green", ls='dashed')
        plt.plot(mass_list, Gas_Fe_over_H_at_all_step_list[2], c="orange", ls='dashed')
        plt.plot(mass_list, Gas_Fe_over_H_at_all_step_list[3], c="red", ls='dashed')
        plt.plot(mass_list, Gas_Fe_over_H_at_all_step_list[4], c="brown", ls='dashed')
    if plot_star == 1:
        plt.plot(mass_list, Star_Fe_over_H_at_all_step_list[4], c="brown")
        plt.plot(mass_list, Star_Fe_over_H_at_all_step_list[3], c="red")
        plt.plot(mass_list, Star_Fe_over_H_at_all_step_list[2], c="orange")
        plt.plot(mass_list, Star_Fe_over_H_at_all_step_list[1], c="green")
        plt.plot(mass_list, Star_Fe_over_H_at_all_step_list[0], c="blue")
    if plot_mid == 1:
        mid0 = []
        mid1 = []
        mid2 = []
        mid3 = []
        mid4 = []
        i = 0
        while i < len(Gas_Fe_over_H_at_all_step_list[0]):
            mid0.append((Gas_Fe_over_H_at_all_step_list[0][i] + Star_Fe_over_H_at_all_step_list[0][i]) / 2)
            mid1.append((Gas_Fe_over_H_at_all_step_list[1][i] + Star_Fe_over_H_at_all_step_list[1][i]) / 2)
            mid2.append((Gas_Fe_over_H_at_all_step_list[2][i] + Star_Fe_over_H_at_all_step_list[2][i]) / 2)
            mid3.append((Gas_Fe_over_H_at_all_step_list[3][i] + Star_Fe_over_H_at_all_step_list[3][i]) / 2)
            mid4.append((Gas_Fe_over_H_at_all_step_list[4][i] + Star_Fe_over_H_at_all_step_list[4][i]) / 2)
            (i) = (i + 1)
        plt.plot(mass_list, mid0, c="blue", ls='dotted')
        plt.plot(mass_list, mid1, c="green", ls='dotted')
        plt.plot(mass_list, mid2, c="orange", ls='dotted')
        plt.plot(mass_list, mid3, c="red", ls='dotted')
        plt.plot(mass_list, mid4, c="brown", ls='dotted')

    if x_axis == "dynamical":
        plt.xlabel(r'log$_{10}$(galaxy dynamical mass) [M$_\odot$]')
    elif x_axis == "luminous":
        plt.xlabel(r'log$_{10}$(galaxy luminous mass) [M$_\odot$]')
    elif x_axis == "LumDyn":
        plt.xlabel(r'log$_{10}$(\sigma) [km/s]')
    elif x_axis == "SFR":
        plt.xlabel(r'log$_{10}$(gwSFR) [M$_\odot$/yr]')
    plt.ylabel('[Fe/H]')
    # plt.xlim(6.4, 1.01 * log_time_axis[-1])
    # plt.ylim(-3, 0.5)
    plt.tight_layout()
    plt.legend(prop={'size': 10}, loc='best')

    ##########################
    ####   Plot [Mg/H]   #####
    ##########################

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(4, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot([], [], c="k", ls='dashed', label='gas')
    plt.plot([], [], c="k", label='stellar')
    plt.plot([], [], c="blue", label='50 Myr')
    plt.plot([], [], c="green", label='100 Myr')
    plt.plot([], [], c="orange", label='500 Myr')
    plt.plot([], [], c="red", label='1 Gyr')
    plt.plot([], [], c="brown", label='10 Gyr')

    plt.plot(mass_list, [0] * len(mass_list), ls='dotted', c='0.5', lw=3,
             label='Solar Value')
    if plot_gas == 1:
        plt.plot(mass_list, Gas_Mg_over_H_at_all_step_list[0], c="blue", ls='dashed')
        plt.plot(mass_list, Gas_Mg_over_H_at_all_step_list[1], c="green", ls='dashed')
        plt.plot(mass_list, Gas_Mg_over_H_at_all_step_list[2], c="orange", ls='dashed')
        plt.plot(mass_list, Gas_Mg_over_H_at_all_step_list[3], c="red", ls='dashed')
        plt.plot(mass_list, Gas_Mg_over_H_at_all_step_list[4], c="brown", ls='dashed')
    if plot_star == 1:
        plt.plot(mass_list, Star_Mg_over_H_at_all_step_list[4], c="brown")
        plt.plot(mass_list, Star_Mg_over_H_at_all_step_list[3], c="red")
        plt.plot(mass_list, Star_Mg_over_H_at_all_step_list[2], c="orange")
        plt.plot(mass_list, Star_Mg_over_H_at_all_step_list[1], c="green")
        plt.plot(mass_list, Star_Mg_over_H_at_all_step_list[0], c="blue")
    if plot_mid == 1:
        mid0 = []
        mid1 = []
        mid2 = []
        mid3 = []
        mid4 = []
        i = 0
        while i < len(Gas_Mg_over_H_at_all_step_list[0]):
            mid0.append((Gas_Mg_over_H_at_all_step_list[0][i] + Star_Mg_over_H_at_all_step_list[0][i]) / 2)
            mid1.append((Gas_Mg_over_H_at_all_step_list[1][i] + Star_Mg_over_H_at_all_step_list[1][i]) / 2)
            mid2.append((Gas_Mg_over_H_at_all_step_list[2][i] + Star_Mg_over_H_at_all_step_list[2][i]) / 2)
            mid3.append((Gas_Mg_over_H_at_all_step_list[3][i] + Star_Mg_over_H_at_all_step_list[3][i]) / 2)
            mid4.append((Gas_Mg_over_H_at_all_step_list[4][i] + Star_Mg_over_H_at_all_step_list[4][i]) / 2)
            (i) = (i + 1)
        plt.plot(mass_list, mid0, c="blue", ls='dotted')
        plt.plot(mass_list, mid1, c="green", ls='dotted')
        plt.plot(mass_list, mid2, c="orange", ls='dotted')
        plt.plot(mass_list, mid3, c="red", ls='dotted')
        plt.plot(mass_list, mid4, c="brown", ls='dotted')

    if x_axis == "dynamical":
        plt.xlabel(r'log$_{10}$(galaxy dynamical mass) [M$_\odot$]')
    elif x_axis == "luminous":
        plt.xlabel(r'log$_{10}$(galaxy luminous mass) [M$_\odot$]')
    elif x_axis == "LumDyn":
        plt.xlabel(r'log$_{10}$(\sigma) [km/s]')
    elif x_axis == "SFR":
        plt.xlabel(r'log$_{10}$(gwSFR) [M$_\odot$/yr]')
    plt.ylabel('[Mg/H]')
    # plt.xlim(6.4, 1.01 * log_time_axis[-1])
    # plt.ylim(-3, 0.5)
    plt.tight_layout()
    plt.legend(prop={'size': 10}, loc='best')

    ##########################
    ####   Plot [Mg/Fe]   #####
    ##########################

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(5, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot([], [], c="k", ls='dashed', label='gas')
    plt.plot([], [], c="k", label='stellar')
    plt.plot([], [], c="blue", label='50 Myr')
    plt.plot([], [], c="green", label='100 Myr')
    plt.plot([], [], c="orange", label='500 Myr')
    plt.plot([], [], c="red", label='1 Gyr')
    plt.plot([], [], c="brown", label='10 Gyr')

    plt.plot(mass_list, [0] * len(mass_list), ls='dotted', c='0.5', lw=3, label='Solar Value')

    plt.plot([1.94, 2.48], [0.05, 0.22], ls='dashed', c='k', lw=1, label='Conroy 2014')  # Conroy 2014 780-33
    plt.plot([1.94, 2.48], [0.12, 0.27], ls='dashed', c='k', lw=1, label='Graves 2008')  # Graves & Schiavon 2008
    plt.plot([1.8, 2.48], [0.13, 0.33], ls='dashed', c='k', lw=1, label='Johansson 2012')  # Johansson 2012
    plt.plot([1.5, 2, 2.48], [-0.24, 0.1, 0.34], ls='dashed', c='k', lw=1, label='Recchi 2009')  # Recchi 2009

    if plot_gas == 1:
        plt.plot(mass_list, Gas_Mg_over_Fe_at_all_step_list[0], c="blue", ls='dashed')
        plt.plot(mass_list, Gas_Mg_over_Fe_at_all_step_list[1], c="green", ls='dashed')
        plt.plot(mass_list, Gas_Mg_over_Fe_at_all_step_list[2], c="orange", ls='dashed')
        plt.plot(mass_list, Gas_Mg_over_Fe_at_all_step_list[3], c="red", ls='dashed')
        plt.plot(mass_list, Gas_Mg_over_Fe_at_all_step_list[4], c="brown", ls='dashed')
    if plot_star == 1:
        plt.plot(mass_list, Star_Mg_over_Fe_at_all_step_list[0], c="blue")
        plt.plot(mass_list, Star_Mg_over_Fe_at_all_step_list[1], c="green")
        plt.plot(mass_list, Star_Mg_over_Fe_at_all_step_list[2], c="orange")
        plt.plot(mass_list, Star_Mg_over_Fe_at_all_step_list[3], c="red")
        plt.plot(mass_list, Star_Mg_over_Fe_at_all_step_list[4], c="brown")
    if plot_mid == 1:
        mid0 = []
        mid1 = []
        mid2 = []
        mid3 = []
        mid4 = []
        i = 0
        while i < len(Gas_Mg_over_Fe_at_all_step_list[0]):
            mid0.append((Gas_Mg_over_Fe_at_all_step_list[0][i] + Star_Mg_over_Fe_at_all_step_list[0][i]) / 2)
            mid1.append((Gas_Mg_over_Fe_at_all_step_list[1][i] + Star_Mg_over_Fe_at_all_step_list[1][i]) / 2)
            mid2.append((Gas_Mg_over_Fe_at_all_step_list[2][i] + Star_Mg_over_Fe_at_all_step_list[2][i]) / 2)
            mid3.append((Gas_Mg_over_Fe_at_all_step_list[3][i] + Star_Mg_over_Fe_at_all_step_list[3][i]) / 2)
            mid4.append((Gas_Mg_over_Fe_at_all_step_list[4][i] + Star_Mg_over_Fe_at_all_step_list[4][i]) / 2)
            (i) = (i + 1)
        plt.plot(mass_list, mid0, c="blue", ls='dotted')
        plt.plot(mass_list, mid1, c="green", ls='dotted')
        plt.plot(mass_list, mid2, c="orange", ls='dotted')
        plt.plot(mass_list, mid3, c="red", ls='dotted')
        plt.plot(mass_list, mid4, c="brown", ls='dotted')

    if x_axis == "dynamical":
        plt.xlabel(r'log$_{10}$(galaxy dynamical mass) [M$_\odot$]')
    elif x_axis == "luminous":
        plt.xlabel(r'log$_{10}$(galaxy luminous mass) [M$_\odot$]')
    elif x_axis == "LumDyn":
        plt.xlabel(r'log$_{10}$(\sigma) [km/s]')
    elif x_axis == "SFR":
        plt.xlabel(r'log$_{10}$(gwSFR) [M$_\odot$/yr]')
    plt.ylabel('[Mg/Fe]')
    # plt.xlim(6.4, 1.01 * log_time_axis[-1])
    # plt.ylim(-3, 0.5)
    plt.tight_layout()
    plt.legend(prop={'size': 10}, loc='best')

    ##########################
    ####   Plot [O/Fe]   #####
    ##########################

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(6, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot([], [], c="k", ls='dashed', label='gas')
    plt.plot([], [], c="k", label='stellar')
    plt.plot([], [], c="blue", label='50 Myr')
    plt.plot([], [], c="green", label='100 Myr')
    plt.plot([], [], c="orange", label='500 Myr')
    plt.plot([], [], c="red", label='1 Gyr')
    plt.plot([], [], c="brown", label='10 Gyr')

    plt.plot(mass_list, [0] * len(mass_list), ls='dotted', c='0.5', lw=3, label='Solar Value')

    plt.plot([1.94, 2.48], [0.03, 0.28], ls='dashed', c='k', lw=1, label='Conroy 2014')  # Conroy 2014 780-33
    plt.plot([1.8, 2.4], [0.1, 0.25], ls='dashed', c='k', lw=1, label='Johansson 2012')  # Johansson 2012
    plt.plot([1.5, 2, 2.48], [-0.24, 0.1, 0.34], ls='dashed', c='k', lw=1, label='Recchi 2009')  # Recchi 2009

    if plot_gas == 1:
        plt.plot(mass_list, Gas_O_over_Fe_at_all_step_list[0], c="blue", ls='dashed')
        plt.plot(mass_list, Gas_O_over_Fe_at_all_step_list[1], c="green", ls='dashed')
        plt.plot(mass_list, Gas_O_over_Fe_at_all_step_list[2], c="orange", ls='dashed')
        plt.plot(mass_list, Gas_O_over_Fe_at_all_step_list[3], c="red", ls='dashed')
        plt.plot(mass_list, Gas_O_over_Fe_at_all_step_list[4], c="brown", ls='dashed')
    if plot_star == 1:
        plt.plot(mass_list, Star_O_over_Fe_at_all_step_list[0], c="blue")
        plt.plot(mass_list, Star_O_over_Fe_at_all_step_list[1], c="green")
        plt.plot(mass_list, Star_O_over_Fe_at_all_step_list[2], c="orange")
        plt.plot(mass_list, Star_O_over_Fe_at_all_step_list[3], c="red")
        plt.plot(mass_list, Star_O_over_Fe_at_all_step_list[4], c="brown")
    if plot_mid == 1:
        mid0 = []
        mid1 = []
        mid2 = []
        mid3 = []
        mid4 = []
        i = 0
        while i < len(Gas_O_over_Fe_at_all_step_list[0]):
            mid0.append((Gas_O_over_Fe_at_all_step_list[0][i] + Star_O_over_Fe_at_all_step_list[0][i]) / 2)
            mid1.append((Gas_O_over_Fe_at_all_step_list[1][i] + Star_O_over_Fe_at_all_step_list[1][i]) / 2)
            mid2.append((Gas_O_over_Fe_at_all_step_list[2][i] + Star_O_over_Fe_at_all_step_list[2][i]) / 2)
            mid3.append((Gas_O_over_Fe_at_all_step_list[3][i] + Star_O_over_Fe_at_all_step_list[3][i]) / 2)
            mid4.append((Gas_O_over_Fe_at_all_step_list[4][i] + Star_O_over_Fe_at_all_step_list[4][i]) / 2)
            (i) = (i + 1)
        plt.plot(mass_list, mid0, c="blue", ls='dotted')
        plt.plot(mass_list, mid1, c="green", ls='dotted')
        plt.plot(mass_list, mid2, c="orange", ls='dotted')
        plt.plot(mass_list, mid3, c="red", ls='dotted')
        plt.plot(mass_list, mid4, c="brown", ls='dotted')

    if x_axis == "dynamical":
        plt.xlabel(r'log$_{10}$(galaxy dynamical mass) [M$_\odot$]')
    elif x_axis == "luminous":
        plt.xlabel(r'log$_{10}$(galaxy luminous mass) [M$_\odot$]')
    elif x_axis == "LumDyn":
        plt.xlabel(r'log$_{10}$(\sigma) [km/s]')
    elif x_axis == "SFR":
        plt.xlabel(r'log$_{10}$(gwSFR) [M$_\odot$/yr]')
    plt.ylabel('[O/Fe]')
    # plt.xlim(6.4, 1.01 * log_time_axis[-1])
    # plt.ylim(-3, 0.5)
    plt.tight_layout()
    plt.legend(prop={'size': 10}, loc='best')

    ##########################
    ####   Plot [O/H]   #####
    ##########################

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(7, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot([], [], c="k", ls='dashed', label='gas')
    plt.plot([], [], c="k", label='stellar')
    plt.plot([], [], c="blue", label='50 Myr')
    plt.plot([], [], c="green", label='100 Myr')
    plt.plot([], [], c="orange", label='500 Myr')
    plt.plot([], [], c="red", label='1 Gyr')
    plt.plot([], [], c="brown", label='10 Gyr')

    plt.plot(mass_list, [0] * len(mass_list), ls='dotted', c='0.5', lw=3,
             label='Solar Value')
    if plot_gas == 1:
        plt.plot(mass_list, Gas_O_over_H_at_all_step_list[0], c="blue", ls='dashed')
        plt.plot(mass_list, Gas_O_over_H_at_all_step_list[1], c="green", ls='dashed')
        plt.plot(mass_list, Gas_O_over_H_at_all_step_list[2], c="orange", ls='dashed')
        plt.plot(mass_list, Gas_O_over_H_at_all_step_list[3], c="red", ls='dashed')
        plt.plot(mass_list, Gas_O_over_H_at_all_step_list[4], c="brown", ls='dashed')
    if plot_star == 1:
        plt.plot(mass_list, Star_O_over_H_at_all_step_list[4], c="brown")
        plt.plot(mass_list, Star_O_over_H_at_all_step_list[3], c="red")
        plt.plot(mass_list, Star_O_over_H_at_all_step_list[2], c="orange")
        plt.plot(mass_list, Star_O_over_H_at_all_step_list[1], c="green")
        plt.plot(mass_list, Star_O_over_H_at_all_step_list[0], c="blue")
    if plot_mid == 1:
        mid0 = []
        mid1 = []
        mid2 = []
        mid3 = []
        mid4 = []
        i = 0
        while i < len(Gas_O_over_H_at_all_step_list[0]):
            mid0.append((Gas_O_over_H_at_all_step_list[0][i] + Star_O_over_H_at_all_step_list[0][i]) / 2)
            mid1.append((Gas_O_over_H_at_all_step_list[1][i] + Star_O_over_H_at_all_step_list[1][i]) / 2)
            mid2.append((Gas_O_over_H_at_all_step_list[2][i] + Star_O_over_H_at_all_step_list[2][i]) / 2)
            mid3.append((Gas_O_over_H_at_all_step_list[3][i] + Star_O_over_H_at_all_step_list[3][i]) / 2)
            mid4.append((Gas_O_over_H_at_all_step_list[4][i] + Star_O_over_H_at_all_step_list[4][i]) / 2)
            (i) = (i + 1)
        plt.plot(mass_list, mid0, c="blue", ls='dotted')
        plt.plot(mass_list, mid1, c="green", ls='dotted')
        plt.plot(mass_list, mid2, c="orange", ls='dotted')
        plt.plot(mass_list, mid3, c="red", ls='dotted')
        plt.plot(mass_list, mid4, c="brown", ls='dotted')

    if x_axis == "dynamical":
        plt.xlabel(r'log$_{10}$(galaxy dynamical mass) [M$_\odot$]')
    elif x_axis == "luminous":
        plt.xlabel(r'log$_{10}$(galaxy luminous mass) [M$_\odot$]')
    elif x_axis == "LumDyn":
        plt.xlabel(r'log$_{10}$(\sigma) [km/s]')
    elif x_axis == "SFR":
        plt.xlabel(r'log$_{10}$(gwSFR) [M$_\odot$/yr]')
    plt.ylabel('[O/H]')
    # plt.xlim(6.4, 1.01 * log_time_axis[-1])
    # plt.ylim(-3, 0.5)
    plt.tight_layout()
    plt.legend(prop={'size': 10}, loc='best')

    #########################

    plt.show()

    # plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='x-small')
    # plt.rc('ytick', labelsize='x-small')
    # fig = plt.figure(2, figsize=(6, 5.25))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.plot(time, Gas_Fe_over_H, label='Gas [Fe/H]')
    # plt.plot(time, Star_Fe_over_H, label='Star [Fe/H]')
    # plt.plot(time, Gas_Mg_over_H, label='Gas [Mg/H]')
    # plt.plot(time, Star_Mg_over_H, label='Star [Mg/H]')
    # plt.xlabel(r'age [yr]')
    # plt.ylabel('[Metal/H]')
    # # plt.xlim(6.4, 1.01 * log_time_axis[-1])
    # plt.ylim(-3, 0.5)
    # plt.tight_layout()
    # plt.legend(prop={'size': 10}, loc='best')
    # plt.show()

    return


if __name__ == '__main__':
    plot(plot_star=1, plot_gas=1, plot_mid=0, x_axis="LumDyn", SFE=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], log_SFR=[-1, 0, 1, 2, 3, 4], IMF="igimf")
    # x_axis = "dynamical" or "luminous" or "LumDyn" or "SFR"
    # LumDyn: Burstein et al. 1997, sigma = 0.86 * (M_lum)^0.22
