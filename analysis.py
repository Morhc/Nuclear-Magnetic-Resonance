from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
from decimal import Decimal as Dc

plt.rcParams['text.usetex'] = True

""" TOOLS """

#loads in the data
def load(*files):

    loaded_results = {}

    for suffix in files:

        file = os.path.join('data', 'T00'+suffix+'CH1.CSV')

        df = pd.read_csv(file, skiprows=15, header=0)

        information = pd.read_csv(file, nrows=14, header=None, names=['Feature', 'Value', 'x', 'xx'], delimiter=',')

        volts_offset = float(information.Value[information.Feature == 'Vertical Offset'])
        volts_scale = float(information.Value[information.Feature == 'Vertical Scale'])
        time_scale = float(information.Value[information.Feature == 'Horizontal Scale'])

        df['TIME'] = df['TIME'] / time_scale

        df['CH1'] = df['CH1'] / volts_scale + volts_offset

        df['CH1 Peak Detect'] = df['CH1 Peak Detect'] / volts_scale + volts_offset

        loaded_results[suffix] = df

    return loaded_results

#smooths the data
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#finds the closest x value index
def closest(x, data): return np.argmin(np.abs(np.asarray(data)-x))

#finds the peak value index
def peak(data): return np.argmax(np.asarray(data))

""" TOOLS """


""" CURVE FITTING """

#B4
def irs(tau, T1): return 1 - 2*np.exp(-tau/T1)

#B5
def decay(t, M, T2): return M*np.exp(-t/T2)

#straight line
def line(x, m, b): return m*x + b

""" CURVE FITTING """

""" EXPERIMENTS """

def B1(data, savename):

    specs = [d for d in data.values()]

    fig, axes = plt.subplots(1,3,figsize=(8,4))

    fig.suptitle('Voltage over Time')

    for i in range(0,3):

        j = closest(0, specs[i]['TIME'].to_numpy())

        time, voltage = specs[i]['TIME'].to_numpy()[j+50:-50], smooth(specs[i]['CH1 Peak Detect'].to_numpy(), 30)[j+50:-50]

        voltage -= np.median(voltage[-100:])

        axes[i].plot(time, voltage, color='black')
        axes[i].set_title(f'{90*(i+1)}' + r'$^o$ ' + 'Pulse Width')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Smoothed Voltage (V)')

        axes[i].set_ylim(-2.5, 5)


    plt.tight_layout()

    savepath = os.path.join('results', 'plots', 'b1_plot.png')
    plt.savefig(savepath)
    plt.close('all')

def B2(data, savename):

    file =[jj for jj in data.keys()][0]

    voltage = smooth(data[file]['CH1 Peak Detect'].to_numpy(), 30)
    time = data[file]['TIME'].to_numpy()

    j = closest(0, time)
    l, i = voltage.size, j + peak(voltage[j:])

    fig, axes = plt.subplots(1,2, figsize=(8,4))


    plt.suptitle(r'Determining T$_2$$^*$ for '  + savename.split('_')[1][0].upper() + savename.split('_')[1][1:])

    #1/e
    f = 0.4

    trans = voltage[i:i+int(f*l)] - np.median(voltage[i+int(f*l)-100:i+int(f*l)])

    M0 = np.max(trans)

    axes[1].plot(time[i:i+int(f*l)], trans/M0, color='black', zorder=2)

    k = closest(1/np.exp(1), trans/M0)
    r = closest(1/3, trans/M0)

    axes[1].axhline(1/np.exp(1), color='red', linestyle='--',zorder=1)
    axes[1].axvline(time[i+k], color='red', linestyle='--', zorder=1)
    axes[1].scatter(time[i+k], 1/np.exp(1), label=r'T$_2$$^*$ = ' +  f'{round(time[i+k],2)}s', zorder=4, color='red', edgecolor='black')

    axes[1].axhline(1/3, color='green', linestyle='--',zorder=1)
    axes[1].axvline(time[i+r], color='green', linestyle='--', zorder=1)
    axes[1].scatter(time[i+r], 1/3, label=r'T$_2$$^*$ = ' + f'{round(time[i+r],2)}s', zorder=4, color='green', edgecolor='black')


    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Normalized Voltage')
    axes[1].set_title(r'Determination by 1/$e$')
    axes[1].legend()

    #Curve fitting
    f = 0.7

    ttt, vvv = time[i:i+int(f*l)], voltage[i:i+int(f*l)] - np.median(voltage[i+int(f*l)-100:i+int(f*l)])

    q = closest((np.max(vvv)+np.min(vvv)) / 2, vvv)

    popt, pcov = curve_fit(f=line, xdata=np.log(ttt)[q-100:q+100], ydata=vvv[q-100:q+100], p0=[time[i+k], -1])

    fity = line(np.log(ttt), *popt)

    axes[0].semilogx(ttt, vvv, color='black', zorder=2)
    axes[0].semilogx(ttt, fity, color='red', zorder=1, label=r'T$_2$$^*$ = ' + f'{round(-popt[0], 2)}' + r'$\pm$' + '%.2e' % Dc(np.sqrt(np.diag(pcov))[0]) + 's')

    axes[0].set_ylim(np.min(vvv) - 0.1, 1.1*np.max(vvv))

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Smoothed Voltage (V)')
    axes[0].set_title('Determination by Slope')
    axes[0].legend()


    plt.tight_layout()

    savepath = os.path.join('results', 'plots', savename+'.png')
    plt.savefig(savepath)
    plt.close('all')

    with open(os.path.join('results', savename.split('_')[1] + '_T2star.csv'), mode='w', encoding = 'utf-8') as f:
        f.writelines('1/e,slope,slope error\n')
        f.writelines(f'{time[i+k]},{-popt[0]},{np.sqrt(np.diag(pcov))[0]}')

    #1/e and slope
    return [time[i+k], -popt[0], np.sqrt(np.diag(pcov))[0]]

def B3(data, T2stars, savename):

    fig, axes = plt.subplots(1,2, figsize=(8,4))

    fig.suptitle(r'Determining $T_1$ for ' + savename.split('_')[1][0].upper() + savename.split('_')[1][1:])

    for nnn, (file, spec) in enumerate(zip(data.keys(), data.values())):

        j = closest(0, spec['TIME'].to_numpy())

        if nnn == 0: axes[nnn].set_title(r'X-Y Plane Verifying $T_2$$^*$')
        if nnn == 1: axes[nnn].set_title(r'Z Plane Determining $T_1$')
        axes[nnn].set_xlabel('Time (s)')
        axes[nnn].set_ylabel('Smoothed Voltage (V)')

        time = spec['TIME'].to_numpy()
        voltage = smooth(spec['CH1 Peak Detect'].to_numpy(), 10)


        cut = closest(np.max(voltage), voltage) if nnn == 0 else 0

        if nnn == 0 and savename.find('rubber') == -1: cut+=50
        elif nnn == 1 and savename.find('ethanol') == -1: cut+=0

        time, voltage = time[j+cut:-30], voltage[j+cut:-30]

        voltage -= np.median(voltage[-100:])

        axes[nnn].plot(time, voltage, color='black',zorder=2)


        if nnn == 0:
            axes[nnn].axvline(5*T2stars[0], color='blue', linestyle='--', zorder=1, label=r'1/$e$')
            axes[nnn].axvline(5*T2stars[1], color='green', linestyle='--', zorder=1, label='Slope')

        elif nnn == 1:

            first, second = peak(voltage[:500]), peak(voltage)

            tau = time[second] - time[first]
            T1 = tau/np.log(voltage[second]/voltage[first])

            plt.scatter([time[first], time[second]], [voltage[first], voltage[second]], marker='x', color='r', zorder=3, label=r'$\tau$ = ' + f'{round(tau, 2)}s')
            plt.scatter(7, voltage[first], color='white', zorder=1, label='$T_1$ = ' + f'{round(T1,2)}s')


        axes[nnn].legend()


    plt.tight_layout()

    savepath = os.path.join('results', 'plots', savename+'.png')
    plt.savefig(savepath)
    plt.close('all')

def B4(data, savename):

    taus, volts = [], []

    for file, spec in zip(data.keys(), data.values()):

        time, voltage = spec['TIME'].to_numpy(), smooth(spec['CH1 Peak Detect'].to_numpy(), 1)
        j, k = closest(0.1, time), closest(8.5, time)
        time, voltage = time[j:k], voltage[j:k]

        adjust = np.median(voltage[300:400]) if file == '19' else np.median(voltage[-100:])

        voltage -= adjust


        if file in ['14', '15']:
            pt = peak(np.abs(voltage[voltage > -3]))

        elif file in ['20', '22']:
            pt1 = peak(-voltage)
            pt = pt1+peak(-voltage[pt1+20:]) + 20

            if file == '20': pt += 8

        elif file == '25':
            pt = peak(voltage[voltage<4])

        elif file in ['27', '28']:
            pt = peak(-voltage)

            if file == '28': pt += 4

        else: pt = peak(np.abs(voltage)) if np.max(voltage) - np.median(voltage) < 0.5 else peak(voltage)


        if file in ['17', '29', '30', '33', '34']: pass
        elif file not in ['16', '23']:
            taus.append(time[pt])
            volts.append(voltage[pt])
        else:
            taus.append(time[pt])
            volts.append(0)


        """
        plt.plot(time, voltage, color='black', zorder=1)
        plt.scatter(time[pt], voltage[pt], marker='x', color='red', zorder=2)

        plt.title('Voltage vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Smoothed Voltage (V)')

        plt.tight_layout()

        savepath = os.path.join('results', 'plots', savename+'_'+file+'.png')
        plt.savefig(savepath)
        plt.close('all')
        """

    fig, axes = plt.subplots(1,2, figsize=(8,4))

    fig.suptitle(r'Determining $T_1$ for ' + savename.split('_')[1][0].upper() + savename.split('_')[1][1:])

    axes[0].set_title('Zero-Crossing Method')
    axes[0].set_ylabel('Normalized Peaks')
    axes[0].set_xlabel(r'$\tau$ (s)')

    taus, volts = np.asarray(taus), (np.asarray(volts)/np.max(volts))


    popt, pcov = curve_fit(f=irs, xdata=taus, ydata=volts, p0=[0.5])
    testx = np.linspace(0,8,300)
    testirs = irs(testx, *popt)

    axes[0].axhline(0, color='grey', linestyle='--', zorder=1)

    if np.sum(volts == 0) == 1: axes[0].axvline(taus[volts == 0], color='green', linestyle='--', zorder=2, label=r'$T_1$ = ' + f'{round((taus[volts == 0][0])/np.log(2),2)}s [Peak]')
    axes[0].axvline(testx[closest(0, testirs)], color='red', linestyle='--', zorder=2, label=r'$T_1$ = ' + f'{round(testx[closest(0, testirs)]/np.log(2),2)}s [Curve]')

    axes[0].plot(testx, testirs, color='black', zorder=3)
    axes[0].scatter(taus, volts, edgecolor='black', facecolor='white', zorder=4, label='Initial Mag. Peak')

    axes[0].legend()


    axes[1].set_title('Slope Method')
    axes[1].set_ylabel('Normalized Peaks')
    axes[1].set_xlabel(r'$\tau$ (s)')

    taus, volts = np.asarray(taus), (np.asarray(volts)/np.max(volts))


    popt, pcov = curve_fit(f=line, xdata=np.log(taus), ydata=volts, p0=[0.05, - np.log(2)])
    testx = np.linspace(0,4,300)
    testline = line(testx, *popt)

    axes[1].axhline(0, color='grey', linestyle='--', zorder=1)

    axes[1].plot(testx, testline, color='black', zorder=3, label=r'$T_1$ = ' + f'{round((testx[closest(1, testline)])/np.log(2), 2)}s')
    axes[1].scatter(np.log(taus), volts, edgecolor='black', facecolor='white', zorder=4, label='Initial Mag. Peak')


    axes[1].legend()



    plt.tight_layout()

    savepath = os.path.join('results', 'plots', savename+'.png')
    plt.savefig(savepath)
    plt.close('all')

def B5(data, savename):

    tau1, tau2, tau3, peak_val = [], [], [], []

    for file, spec in zip(data.keys(), data.values()):


        time, voltage = spec['TIME'].to_numpy(), smooth(spec['CH1 Peak Detect'].to_numpy(), 30)

        j, k = closest(-0.1, time), closest(8.5, time)

        time, voltage = time[j:k], voltage[j:k]
        voltage -= np.median(voltage[-100:])

        push = 1000 if savename.find('rubber') == -1 else 2000

        one, two = peak(voltage[:500]), peak(voltage[push:])
        mid = int((two+push+one)/2)

        way1, way2, way3 = time[mid] - time[one], time[two+push] - time[mid], (time[two+push] + time[one])/2

        tau1.append(way1)
        tau2.append(way2)
        tau3.append(way3)

        peak_val.append(-voltage[two+push])



        plt.scatter([time[one], time[two+push]], [voltage[one], voltage[two+push]], marker='x', color='red', zorder=2)
        plt.scatter(time[mid], voltage[mid], marker='x', color='yellow', zorder=3)


        plt.plot(time, voltage, color='black',zorder=1)

        plt.title('Voltage vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Smoothed Voltage (V)')

        plt.tight_layout()

        savepath = os.path.join('results', 'plots', savename+'_'+file+'.png')
        plt.savefig(savepath)
        plt.close('all')


    t1, t2, t3, peak_val = 2*np.asarray(tau1), 2*np.asarray(tau2), 2*np.asarray(tau3), np.asarray(peak_val) / np.max(peak_val)


    fig, axes = plt.subplots(1,2, figsize=(8,4), sharex=True)
    plt.subplots_adjust(wspace=0)

    fig.suptitle(r'Determining $T_2$ for '+ savename.split('_')[1][0].upper() + savename.split('_')[1][1:])

    axes[0].set_title('Unadjusted Results')
    axes[1].set_title('Adjusted Results')

    axes[0].set_ylabel('Hanh Echo Peaks')
    for i in range(2):
        axes[i].set_xlabel(r'$t = 2 \tau$ (s)')

        if savename.find('rubber') != -1: filter = [True, True, True, True, True] if i == 0 else [False, True, True, True, True]
        elif savename.find('water') != -1: filter = [True, True, True, True, True] if i == 0 else [True, False, True, True, True]
        else: filter = [True, True, True, True, True] if i == 0 else [True, False, True, True, False]
        t1, t2, t3, peak_val = t1[filter], t2[filter], t3[filter], peak_val[filter]

        popt1, pcov1 = curve_fit(f=decay, xdata=t1, ydata=peak_val, p0=[2,10])
        popt2, pcov2 = curve_fit(f=decay, xdata=t2, ydata=peak_val, p0=[2,10])
        popt3, pcov3 = curve_fit(f=decay, xdata=t3, ydata=peak_val, p0=[2,10])

        axes[i].scatter(t1, peak_val, color='blue', edgecolor='black', zorder = 2)
        axes[i].scatter(t2, peak_val, color='orange', edgecolor='black', zorder = 2)
        axes[i].scatter(t3, peak_val, color='green', edgecolor='black', zorder = 2)


        xtest = np.linspace(2.5,8,100)
        axes[i].plot(xtest, decay(xtest, *popt1), color='blue', zorder=1, label=r'$T_1$ = ' + f'{round(popt1[1], 2)}' + r'$\pm$' + f'{round(np.diag(np.sqrt(pcov1))[1], 2)}')
        axes[i].plot(xtest, decay(xtest, *popt2), color='orange', zorder=1, label=r'$T_1$ = ' + f'{round(popt2[1], 2)}' + r'$\pm$' + f'{round(np.diag(np.sqrt(pcov2))[1], 2)}')
        axes[i].plot(xtest, decay(xtest, *popt3), color='green', zorder=1, label=r'$T_1$ =' + f'{round(popt3[1], 2)}' + r'$\pm$' + f'{round(np.diag(np.sqrt(pcov3))[1], 2)}')

        axes[i].legend()

    plt.tight_layout()

    savepath = os.path.join('results', 'plots', savename+'.png')
    plt.savefig(savepath)
    plt.close('all')

def B6(data, savename):

    specs = [d for d in data.values()]

    fig, axes = plt.subplots(1,3,figsize=(8,4))

    fig.suptitle('Voltage over Time')

    axes[0].set_title('Liquid Water')
    axes[1].set_title('Partially Frozen Water')
    axes[2].set_title('Fully Frozen Water')

    for i in range(0,3):

        j = closest(0, specs[i]['TIME'].to_numpy())

        time, voltage = specs[i]['TIME'].to_numpy()[j+50:-50], smooth(specs[i]['CH1 Peak Detect'].to_numpy(), 30)[j+50:-50]

        voltage -= np.median(voltage[-100:])

        axes[i].plot(time, voltage, color='black')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Smoothed Voltage (V)')

        axes[i].set_ylim(-0.5,4)


    plt.tight_layout()

    savepath = os.path.join('results', 'plots', 'b6_plot.png')
    plt.savefig(savepath)
    plt.close('all')

""" EXPERIMENTS """

def main():

    B1(load('00','01', '02'), 'b1_water')

    wT2star = B2(load('03'), 'b2_water')
    rT2star = B2(load('04'), 'b2_rubber')
    eT2star = B2(load('05'), 'b2_ethanol')

    B3(load('07', '12'), wT2star, 'b3_water')
    B3(load('08', '13'), rT2star, 'b3_rubber')
    B3(load('10', '11'), eT2star, 'b3_ethanol')

    B4(load('14', '15', '16', '17', '18', '19'), 'b4_water')
    B4(load('20', '22', '23', '24', '25', '26'), 'b4_rubber')
    B4(load('27', '28', '29', '30', '34', '33', '35'), 'b4_ethanol')

    B5(load('36', '37', '38', '39', '40'), 'b5_water')
    B5(load('41', '42', '43', '44', '45'), 'b5_rubber')
    B5(load('46', '47', '48', '49', '50'), 'b5_ethanol')

    B6(load('51', '53', '52'), 'b6_water')

if __name__ == '__main__':
    main()
