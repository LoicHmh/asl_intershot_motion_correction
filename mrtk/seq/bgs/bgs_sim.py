import numpy as np
import matplotlib.pyplot as plt


def OptimalTauDoubleInversion(T1opt, TI):
    """
    This function calculates the theoretically ideal tau1 and tau2 values
    used to null species with T1 = T1opt and 2*T1opt (from Gunther et al, MRM
    2005)
    """

    Tau1 = TI + 2 * T1opt * np.log(0.75 + 0.25 * np.exp(-0.5 * TI / T1opt))
    Tau2 = TI + 2 * T1opt * np.log(0.25 + 0.75 * np.exp(-0.5 * TI / T1opt))
    
    # Ensure Tau1 is less than Tau2
    if Tau1 > Tau2:
        Tau1, Tau2 = Tau2, Tau1
    return Tau1, Tau2


def OptimalTausFourBGSPulsesMaleki(ts):
    # Define the coefficients
    p = np.array([[-1.385e-005, 0.08497, -7.52], 
                  [-4.649e-005, 0.3216, -21.42], 
                  [-6.983e-005, 0.6480, -28.95], 
                  [-4.308e-005, 0.9219, -19.40]])

    # Run the calculation
    taus = p[:, 0] * ts**2 + p[:, 1] * ts + p[:, 2]

    # Note that this equation returns the time between each pulse and the
    # readout, so for our definition of time between the presat and the pulse
    # we need to subtract from ts and sort
    taus = np.sort(ts - taus)
    return taus



def CalcFinalMzForMultipleBGSPulses(T1, tauvec, TI, AlphaInv=1):
    # Calculate the final Mz for a given T1 species given pre-saturation and
    # N inversion pulses at times defined by the tau vector and final time TI

    # Assume we start with an efficient presat
    Mz = np.zeros_like(T1)  # Allow for multiple T1 values to be dealt with simultaneously
    t = 0

    # Loop through the inversion pulses
    for ii in range(len(tauvec)):
        tnow = tauvec[ii]
        telapsed = tnow - t  # Time since the last pulse

        # Calculate Mz changed due to T1 relaxation until just before the BGS pulse
        Mz = 1 - (1 - Mz) * np.exp(-telapsed / T1)

        # After the first pulse this is negated with efficiency AlphaInv
        Mz = Mz - 2 * Mz * AlphaInv

        # Update t
        t = tnow

    # T1 decay until TI
    FinalMz = 1 - (1 - Mz) * np.exp(-(TI - t) / T1)
    return FinalMz


def PlotMzEvolutionForT1(T1, total_time, tauvec, AlphaInv=1):
    """
    Plot the Mz evolution over time for a given T1.
    
    Parameters:
    - T1: T1 relaxation time in ms
    - total_time: Total duration of simulation in ms
    - tauvec: List of times (ms) at which BGS pulses are applied
    - AlphaInv: Efficiency of inversion pulses
    """

    # print(tauvec)

    time_points = np.linspace(0, total_time, 5000)
    Mz = np.zeros_like(time_points)
    t_last = 0
    Mz_last = 0
    i_inverse_pulse = 0
    for i, t in enumerate(time_points):
        if i_inverse_pulse < len(tauvec) and t >= tauvec[i_inverse_pulse] :  # Apply BGS pulse
            # print("inverse!")
            i_inverse_pulse += 1
            Mz[i] = Mz[i-1] - 2 * Mz[i-1] * AlphaInv  # Inversion
            t_last = t
            Mz_last = Mz[i]
        else:
            # T1 relaxation between pulses
            elapsed = t - t_last
            
            Mz[i] = 1 - (1 - Mz_last) * np.exp(-elapsed / T1) if i > 0 else 0
        

    return time_points, Mz


def main():
    # Compare two pulse (Gunther et al) and four pulse (Maleki et al) BGS schemes

    # Set up parameters
    T1s = np.arange(50, 4000, 50)
    AlphaInv = 0.95
    T1Opt = 700
    tau = 1800
    PLD = 1800
    TI = tau + PLD

    # Two pulses
    Tau1, Tau2 = OptimalTauDoubleInversion(T1Opt, TI)
    tauvec_DI = [Tau1, Tau2]
    FinalMz = CalcFinalMzForMultipleBGSPulses(T1s, tauvec_DI, TI, AlphaInv)
    plt.figure()
    plt.plot(T1s, FinalMz * 100, linewidth=2)

    # Four pulses
    tauvec_4I = OptimalTausFourBGSPulsesMaleki(TI)
    FinalMz = CalcFinalMzForMultipleBGSPulses(T1s, tauvec_4I, TI, 1)
    plt.plot(T1s, FinalMz * 100, linewidth=2)
    plt.grid(True)
    plt.legend(['Double Inversion', 'Four Inversion'])
    plt.xlabel('T1/ms')
    plt.ylabel('Final Mz/%')
    plt.show()


    # Plot Mz evolution
    plt.figure(figsize=(8, 5))
    for (tissue, T1) in [('CSF', 4000), ('GM', 1800), ('WM', 1200), ('Blood', 1650)]:

        time_points, Mz = PlotMzEvolutionForT1(T1=T1, total_time=TI, tauvec=tauvec_4I, AlphaInv=1)
        plt.plot(time_points, Mz, label=f"{tissue}(T1={T1}ms)", linewidth=2)

        # time_points, Mz = PlotMzEvolutionForT1(T1=T1, total_time=TI, tauvec=tauvec_DI, AlphaInv=AlphaInv)
        # plt.plot(time_points, Mz, '--',  label=f"{tissue}(T1 = {T1} ms)", linewidth=2)

        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel("Time/ms")

    plt.ylabel("Mz")
    plt.title(f"Mz Evolution for Four Inversion BGS with TI={TI} ms")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    for (tissue, T1) in [('CSF', 4000), ('GM', 1800), ('WM', 1200), ('Blood', 1650)]:

        # time_points, Mz = PlotMzEvolutionForT1(T1=T1, total_time=TI, tauvec=tauvec_4I, AlphaInv=1)
        # plt.plot(time_points, Mz, label=f"{tissue}(T1 = {T1} ms)", linewidth=2)

        time_points, Mz = PlotMzEvolutionForT1(T1=T1, total_time=TI, tauvec=tauvec_DI, AlphaInv=AlphaInv)
        plt.plot(time_points, Mz, label=f"{tissue}(T1={T1}ms)", linewidth=2)

        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel("Time (ms)")

    plt.ylabel("Mz/%")
    plt.title(f"Mz Evolution for Double Inversion BGS with T1Opt={T1Opt} ms")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()