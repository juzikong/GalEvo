import time
import numpy as np
import math
import matplotlib.pyplot as plt


def function_read_file(yield_table_name):

    ####################
    ### read in file ###
    ####################
    if yield_table_name == "portinari98":
        file_yield = open(
            'yield_tables/agb_and_massive_stars_portinari98_marigo01_gce_totalyields.txt', 'r')
        # Use net yields of Portinari and Marigo
        # Net yields with masses up to 7Msun are from Marigo, above those of Portinari are taken.
        # Only isotopes are selected which are available in both yield sets and go up to Fe.
        # Initial masses go from the lowest mass available up to 100Msun.
        # Yield set ID M01P98 in Ritter et al. 2017.
        # References: Marigo et al. 2001, http://ukads.nottingham.ac.uk/abs/2001A%26A...370..194M
        #		      Portinari et al. 1998, http://ukads.nottingham.ac.uk/abs/1998A%26A...334..505P
        data = file_yield.readlines()
        file_yield.close()
    elif yield_table_name == "WW95":
        file_yield = open(
            'yield_tables/massive_stars_WW95_totalyields.txt', 'r')
        # Use net yields of Woosley S. E., Weaver T. A., 1995, ApJS, 101, 181 (WW95)
        # Use WW95 model B which has the highest [Mg/Fe].
        data = file_yield.readlines()
        file_yield.close()
    elif yield_table_name == "marigo01":
        file_yield = open(
            'yield_tables/agb_marigo01_totalyields.txt', 'r')
        data = file_yield.readlines()
        file_yield.close()


    ###########################
    ### extract information ###
    ###########################
    #
    H_relative_line_number = function_get_element_line_number(data, 'H-1')
    He_relative_line_number = function_get_element_line_number(data, 'He-4')
    C_relative_line_number = function_get_element_line_number(data, 'C-12')
    N_relative_line_number = function_get_element_line_number(data, 'N-14')
    O_relative_line_number = function_get_element_line_number(data, 'O-16')
    Ne_relative_line_number = function_get_element_line_number(data, 'Ne-20')
    Mg_relative_line_number = function_get_element_line_number(data, 'Mg-24')
    Si_relative_line_number = function_get_element_line_number(data, 'Si-28')
    S_relative_line_number = function_get_element_line_number(data, 'S-32')
    Ca_relative_line_number = function_get_element_line_number(data, 'Ca-40')
    Fe_relative_line_number = function_get_element_line_number(data, 'Fe-56')
    #
    global M_list, Z_list, eject_mass_list, H_eject_mass_list, He_eject_mass_list, C_eject_mass_list, \
        N_eject_mass_list, O_eject_mass_list, Ne_eject_mass_list, Mg_eject_mass_list, Si_eject_mass_list, \
        S_eject_mass_list, Ca_eject_mass_list,  Fe_eject_mass_list, Metal_eject_mass_list
    global O_over_Mg_list, Mg_over_Fe_list, Mg_over_H_list, Fe_over_H_list, O_over_H_list, Z_over_H_list, O_over_Fe_list
    #
    i = 0
    while i < len(data):
        line_i = str.split(data[i])
        if line_i[1] == 'Table:':
            line_H = str.split(data[i + H_relative_line_number])
            line_He = str.split(data[i + He_relative_line_number])
            line_C = str.split(data[i + C_relative_line_number])
            line_N = str.split(data[i + N_relative_line_number])
            line_O = str.split(data[i + O_relative_line_number])
            line_Ne = str.split(data[i + Ne_relative_line_number])
            line_Mg = str.split(data[i + Mg_relative_line_number])
            line_Si = str.split(data[i + Si_relative_line_number])
            line_S = str.split(data[i + S_relative_line_number])
            line_Ca = str.split(data[i + Ca_relative_line_number])
            line_Fe = str.split(data[i + Fe_relative_line_number])
            line_Mfinal = str.split(data[i + 2])
            (Z, M) = function_get_Z_M(line_i[2])
            ejecta_mass = round((M - function_get_Mfinal(line_Mfinal[2])), 5)  ####################
            H_mass = function_get_element_mass(line_H[1])
            He_mass = function_get_element_mass(line_He[1])
            C_mass = function_get_element_mass(line_C[1])
            N_mass = function_get_element_mass(line_N[1])
            O_mass = function_get_element_mass(line_O[1])
            Ne_mass = function_get_element_mass(line_Ne[1])
            Mg_mass = function_get_element_mass(line_Mg[1])
            Si_mass = function_get_element_mass(line_Si[1])
            S_mass = function_get_element_mass(line_S[1])
            Ca_mass = function_get_element_mass(line_Ca[1])
            Fe_mass = function_get_element_mass(line_Fe[1])
            H_num = H_mass/1.0079
            O_num = O_mass/15.9994
            Mg_num = Mg_mass/24.305
            Fe_num = Fe_mass/55.845
            O_over_Mg = math.log(O_num/Mg_num, 10) - 8.69 + 7.60
            Mg_over_H = math.log(Mg_num/H_num, 10) - 7.60 + 12
            Fe_over_H = math.log(Fe_num/H_num, 10) - 7.50 + 12
            O_over_H = math.log(O_num/H_num, 10) - 8.69 + 12
            Mg_over_Fe = math.log(Mg_num/Fe_num, 10) - 7.60 + 7.50
            O_over_Fe = math.log(O_num/Fe_num, 10) - 8.69 + 7.50
            Metal_mass = round((ejecta_mass - H_mass - He_mass), 5)  ####################
            if Metal_mass<0:
                print("Warning: Metal_mass=", Metal_mass, "<0")
                print("check stellar yield table with metallicity and mass being:", Z, "&", M)
                Metal_mass = 0
            Z_over_H = math.log(Metal_mass / H_mass, 10) - math.log(0.0134 / 0.7381, 10)
            if len(Z_list) == 0:
                Z_list.append(Z)
                Z_n = 0
                M_list.append([])
                eject_mass_list.append([])
                H_eject_mass_list.append([])
                He_eject_mass_list.append([])
                C_eject_mass_list.append([])
                N_eject_mass_list.append([])
                O_eject_mass_list.append([])
                Ne_eject_mass_list.append([])
                Mg_eject_mass_list.append([])
                Si_eject_mass_list.append([])
                S_eject_mass_list.append([])
                Ca_eject_mass_list.append([])
                Fe_eject_mass_list.append([])
                Metal_eject_mass_list.append([])
                Z_over_H_list.append([])
                O_over_Mg_list.append([])
                Mg_over_Fe_list.append([])
                Mg_over_H_list.append([])
                Fe_over_H_list.append([])
                O_over_H_list.append([])
                O_over_Fe_list.append([])
            if Z != Z_list[-1]:
                Z_list.append(Z)
                Z_n += 1
                M_list.append([])
                eject_mass_list.append([])
                H_eject_mass_list.append([])
                He_eject_mass_list.append([])
                C_eject_mass_list.append([])
                N_eject_mass_list.append([])
                O_eject_mass_list.append([])
                Ne_eject_mass_list.append([])
                Mg_eject_mass_list.append([])
                Si_eject_mass_list.append([])
                S_eject_mass_list.append([])
                Ca_eject_mass_list.append([])
                Fe_eject_mass_list.append([])
                Metal_eject_mass_list.append([])
                O_over_Mg_list.append([])
                Mg_over_Fe_list.append([])
                Mg_over_H_list.append([])
                Fe_over_H_list.append([])
                O_over_H_list.append([])
                Z_over_H_list.append([])
                O_over_Fe_list.append([])
            M_list[Z_n].append(M)
            eject_mass_list[Z_n].append(ejecta_mass)
            H_eject_mass_list[Z_n].append(H_mass)
            He_eject_mass_list[Z_n].append(He_mass)
            C_eject_mass_list[Z_n].append(C_mass)
            N_eject_mass_list[Z_n].append(N_mass)
            O_eject_mass_list[Z_n].append(O_mass)
            Ne_eject_mass_list[Z_n].append(Ne_mass)
            Mg_eject_mass_list[Z_n].append(Mg_mass)
            Si_eject_mass_list[Z_n].append(Si_mass)
            S_eject_mass_list[Z_n].append(S_mass)
            Ca_eject_mass_list[Z_n].append(Ca_mass)
            Fe_eject_mass_list[Z_n].append(Fe_mass)
            Metal_eject_mass_list[Z_n].append(Metal_mass)
            O_over_Mg_list[Z_n].append(O_over_Mg)
            Mg_over_Fe_list[Z_n].append(Mg_over_Fe)
            Mg_over_H_list[Z_n].append(Mg_over_H)
            O_over_H_list[Z_n].append(O_over_H)
            Z_over_H_list[Z_n].append(Z_over_H)
            Fe_over_H_list[Z_n].append(Fe_over_H)
            O_over_Fe_list[Z_n].append(O_over_Fe)
        (i) = (i + 1)

    return

def function_get_Mfinal(Mfinal_string):
    i_end = len(Mfinal_string)
    i = 0
    mass_str = ''
    while i < i_end:
        mass_str += Mfinal_string[i]
        (i) = (i + 1)
    mass = float(mass_str)
    return mass

def function_get_element_mass(element_mass_string):
    i_end = len(element_mass_string)
    i = 1
    mass_str = ''
    while i < i_end:
        mass_str += element_mass_string[i]
        (i) = (i + 1)
    mass = float(mass_str)
    return mass

def function_get_element_line_number(data, element):
    i = 0
    while i < len(data):
        line_i = str.split(data[i])
        if line_i[1] == 'Table:':
            start = i
            j = 0
            while j < 100:
                line_j = str.split(data[j])
                if line_j[0] == '&'+element:
                    end = j
                    element_relative_line_number = j - i
                    break
                (j) = (j+1)
            break
        (i) = (i + 1)
    return element_relative_line_number

def function_get_Z_M(M_Z_string):
    i = 0
    i_M_start = 0
    i_M_end = 0
    i_Z_start = 0
    i_Z_end = 0
    while i < len(M_Z_string):
        if M_Z_string[i] == 'M':
            i_M_start = i+2
        if M_Z_string[i] == ',':
            i_M_end = i
            i_Z_start = i+3
        if M_Z_string[i] == ')':
            i_Z_end = i
        (i) = (i+1)
    i = i_Z_start
    Z_str = ''
    while i < i_Z_end:
        Z_str += M_Z_string[i]
        (i) = (i + 1)
    Z = float(Z_str)
    i = i_M_start
    M_str = ''
    while i < i_M_end:
        M_str += M_Z_string[i]
        (i) = (i + 1)
    M = float(M_str)
    return (Z, M)

def funtion_plot_yields():
    global O_over_Mg_list, Mg_over_Fe_list, Mg_over_H_list, Fe_over_H_list, O_over_H_list, Z_over_H_list, O_over_Fe_list, M_list, Z_list
    j = 0
    while j < len(M_list):
        i = 0
        while i < len(M_list[j]):
            M_list[j][i] = math.log(M_list[j][i], 10)
            (i) = (i+1)
        (j) = (j+1)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(1, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.5, 2.2)
    plt.ylim(0, 2)
    i = 0
    while i < len(M_list):
        plt.plot(M_list[i], O_over_Mg_list[i], label='Z={}'.format(Z_list[i]))
        (i) = (i+1)
    O_mass_eject_SNIa = 0.148  # TNH93 0.148 i99CDD1 0.09, i99CDD2 0.06, i99W7 0.14, ivo12/13 0.09-0.1, t03 0.14, t86 0.13
    Mg_mass_eject_SNIa = 0.009  # TNH93 0.009 i99CDD1 0.0077, i99CDD2 0.0042, i99W7 0.0085, ivo12/13 0.015-0.029, t03 0.013, t86 0.016
    O_num = O_mass_eject_SNIa / 15.9994
    Mg_num = Mg_mass_eject_SNIa / 24.305
    O_over_Mg_SNIa = math.log(O_num / Mg_num, 10) - 8.69 + 7.60
    plt.plot([-0.3, 0.9], [O_over_Mg_SNIa, O_over_Mg_SNIa], ls="--", lw=2, label="SNIa")
    plt.legend(prop={'size': 10}, loc='best')
    plt.xlabel(r'log stellar mass [M$_\odot$]')
    plt.ylabel(r'[O/Mg]')
    plt.tight_layout()


    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(2, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.5, 2.2)
    plt.ylim(-2, 7)
    i = 0
    while i < len(M_list):
        plt.plot(M_list[i], Mg_over_Fe_list[i], label='Z={}'.format(Z_list[i]))
        (i) = (i+1)
    Mg_mass_eject_SNIa = 0.009  # TNH93 0.148 i99CDD1 0.09, i99CDD2 0.06, i99W7 0.14, ivo12/13 0.09-0.1, t03 0.14, t86 0.13
    Fe_mass_eject_SNIa = 0.372  #0.63 # Recchi2009 halfed to 0.372  # TNH93 0.744 i99CDD1 0.56, i99CDD2 0.76, i99W7 0.63, ivo12/13 0.62-0.67, t03 0.74, t86 0.63
    Mg_num = Mg_mass_eject_SNIa / 24.305
    Fe_num = Fe_mass_eject_SNIa / 55.845
    Mg_over_Fe_SNIa = math.log(Mg_num / Fe_num, 10) - 7.60 + 7.50
    plt.plot([-0.3, 0.9], [Mg_over_Fe_SNIa, Mg_over_Fe_SNIa], ls="--", lw=2, label="SNIa")
    plt.plot([-2, 3], [0, 0], lw=0.1)
    plt.legend(prop={'size': 10}, loc='best')
    plt.xlabel(r'log stellar mass [M$_\odot$]')
    plt.ylabel(r'[Mg/Fe]')
    plt.tight_layout()


    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(3, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.5, 2.2)
    plt.ylim(-2, 7)
    i = 0
    while i < len(M_list):
        plt.plot(M_list[i], O_over_Fe_list[i], label='Z={}'.format(Z_list[i]))
        (i) = (i+1)
    O_over_Fe_SNIa = math.log(O_num / Fe_num, 10) - 7.60 + 7.50
    plt.plot([-0.3, 0.9], [O_over_Fe_SNIa, O_over_Fe_SNIa], ls="--", lw=2, label="SNIa")
    plt.legend(prop={'size': 10}, loc='best')
    plt.xlabel(r'log stellar mass [M$_\odot$]')
    plt.ylabel(r'[O/Fe]')
    plt.tight_layout()

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(4, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.5, 2.2)
    plt.ylim(-2, 2)
    i = 0
    while i < len(M_list):
        plt.plot(M_list[i], Mg_over_H_list[i], label='Z={}'.format(Z_list[i]))
        (i) = (i + 1)
    plt.plot([-2, 3], [0, 0], lw=0.1)
    plt.legend(prop={'size': 10}, loc='best')
    plt.xlabel(r'log stellar mass [M$_\odot$]')
    plt.ylabel(r'[Mg/H]')
    plt.tight_layout()

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(5, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.5, 2.2)
    plt.ylim(-2, 2)
    i = 0
    while i < len(M_list):
        plt.plot(M_list[i], O_over_H_list[i], label='Z={}'.format(Z_list[i]))
        (i) = (i + 1)
    plt.plot([-2, 3], [0, 0], lw=0.1)
    plt.legend(prop={'size': 10}, loc='best')
    plt.xlabel(r'log stellar mass [M$_\odot$]')
    plt.ylabel(r'[O/H]')
    plt.tight_layout()

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(6, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.5, 2.2)
    plt.ylim(-2, 2)
    i = 0
    while i < len(M_list):
        plt.plot(M_list[i], Fe_over_H_list[i], label='Z={}'.format(Z_list[i]))
        (i) = (i + 1)
    plt.plot([-2, 3], [0, 0], lw=0.1)
    plt.legend(prop={'size': 10}, loc='best')
    plt.xlabel(r'log stellar mass [M$_\odot$]')
    plt.ylabel(r'[Fe/H]')
    plt.tight_layout()

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(7, figsize=(6, 5.25))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.5, 2.2)
    plt.ylim(-2, 2)
    i = 0
    while i < len(M_list):
        plt.plot(M_list[i], Z_over_H_list[i], label='Z={}'.format(Z_list[i]))
        (i) = (i + 1)
    plt.plot([-2, 3], [0, 0], lw=0.1)
    plt.legend(prop={'size': 10}, loc='best')
    plt.xlabel(r'log stellar mass [M$_\odot$]')
    plt.ylabel(r'[Z/H]')
    plt.tight_layout()


    plt.show()
    return


if __name__ == '__main__':
    start_time = time.time()
    Z_list = []
    M_list = []
    eject_mass_list = []
    H_eject_mass_list = []
    He_eject_mass_list = []
    C_eject_mass_list = []
    N_eject_mass_list = []
    O_eject_mass_list = []
    Ne_eject_mass_list = []
    Mg_eject_mass_list = []
    Si_eject_mass_list = []
    S_eject_mass_list = []
    Ca_eject_mass_list = []
    Fe_eject_mass_list = []
    Metal_eject_mass_list = []
    O_over_Mg_list = []
    Mg_over_H_list = []
    Fe_over_H_list = []
    O_over_H_list = []
    Z_over_H_list = []
    Mg_over_Fe_list = []
    O_over_Fe_list = []
    yield_table_name = "portinari98" # being "WW95" or "portinari98" or "marigo01"
    function_read_file(yield_table_name)
    funtion_plot_yields()
    print(" - Run time: %s -" % round((time.time() - start_time), 2))