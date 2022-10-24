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

#exponential decay
def decay(t, M0, T2star): return M0*np.exp(-t/T2star)

#straight line
def line(x, m, b): return m*x + b

""" CURVE FITTING """

""" EXPERIMENTS """

def B1(data, savename):

    specs = [d for d in data.values()]

    fig, axes = plt.subplots(3,1,figsize=(5,8))

    fig.suptitle('Voltage vs Time')

    for i in range(0,3):

        j = closest(0, specs[i]['TIME'].to_numpy())

        axes[i].plot(specs[i]['TIME'].to_numpy()[j+50:-50], smooth(specs[i]['CH1 Peak Detect'].to_numpy(), 30)[j+50:-50], color='black')
        axes[i].set_title(f'{90*(i+1)}' + r'$^o$ ' + 'Pulse Width')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Smoothed Voltage (V)')

        axes[i].set_ylim(-8.5, -0.5)


    plt.tight_layout()

    savepath = os.path.join('results', 'plots', 'B1_Plot.png')
    plt.savefig(savepath)
    plt.close('all')

def B2(data, savename):

    for i, (file, spec) in enumerate(zip(data.keys(), data.values())):

        voltage = smooth(spec['CH1 Peak Detect'].to_numpy(), 30)
        time = spec['TIME'].to_numpy()

        j = closest(0, time)
        l, i = voltage.size, j + peak(voltage[j:])

        fig, axes = plt.subplots(1,2, figsize=(8,4))


        plt.suptitle(r'Determining T$_2$$^*$ for '  + savename.split('_')[1][0].upper() + savename.split('_')[1][1:])

        #1/e
        f = 0.4

        trans = voltage[i:i+int(f*l)] - np.min(voltage[i:i+int(f*l)])

        M0 = np.max(trans)

        axes[1].plot(time[i:i+int(f*l)], trans/M0, color='black', zorder=2)

        k = closest(1/np.exp(1), trans/M0)

        axes[1].axhline(1/np.exp(1), color='red', linestyle='--',zorder=1)
        axes[1].axvline(time[i+k], color='red', linestyle='--', zorder=1)
        axes[1].scatter(time[i+k], 1/np.exp(1), label=r'T$_2$$^*$ = ' + '%.2e' % Dc(time[i+k]) + 's', zorder=4, color='red', edgecolor='black')

        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Peak Voltage (V)')
        axes[1].set_title(r'Determination by 1/$e$')
        axes[1].legend()

        #Curve fitting
        f = 0.7

        ttt, vvv = time[i:i+int(f*l)], voltage[i:i+int(f*l)]

        q = closest((np.max(vvv)+np.min(vvv)) / 2, vvv)

        popt, pcov = curve_fit(f=line, xdata=np.log(ttt)[q-100:q+100], ydata=vvv[q-100:q+100], p0=[time[i+k], -1])

        fity = line(np.log(ttt), *popt)

        axes[0].semilogx(ttt, vvv, color='black', zorder=2)
        axes[0].semilogx(ttt, fity, color='red', zorder=1, label=r'T$_2$$^*$ = ' + '%.2e' % Dc(-popt[0]) + r'$\pm$' + '%.2e' % Dc(np.sqrt(np.diag(pcov))[0]) + 's')

        axes[0].set_ylim(1.1*np.min(vvv), 0.8*np.max(vvv))

        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Peak Voltage (V)')
        axes[0].set_title('Determination by Slope')
        axes[0].legend()


        plt.tight_layout()

        savepath = os.path.join('results', 'plots', savename+'_'+file+'.png')
        plt.savefig(savepath)
        plt.close('all')

def B3(data, savename):

    for file, spec in zip(data.keys(), data.values()):





        plt.plot(spec['TIME'].to_numpy(), smooth(spec['CH1 Peak Detect'].to_numpy(), 30), color='black')

        plt.title('Voltage vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Peak Voltage (V)')

        plt.tight_layout()

        savepath = os.path.join('results', 'plots', savename+'_'+file+'.png')
        plt.savefig(savepath)
        plt.close('all')

def B4(data, savename):

    for file, spec in zip(data.keys(), data.values()):





        plt.plot(spec['TIME'].to_numpy(), smooth(spec['CH1 Peak Detect'].to_numpy(), 30), color='black')

        plt.title('Voltage vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Peak Voltage (V)')

        plt.tight_layout()

        savepath = os.path.join('results', 'plots', savename+'_'+file+'.png')
        plt.savefig(savepath)
        plt.close('all')

def B5(data, savename):

    for file, spec in zip(data.keys(), data.values()):





        plt.plot(spec['TIME'].to_numpy(), smooth(spec['CH1 Peak Detect'].to_numpy(), 30), color='black')

        plt.title('Voltage vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Peak Voltage (V)')

        plt.tight_layout()

        savepath = os.path.join('results', 'plots', savename+'_'+file+'.png')
        plt.savefig(savepath)
        plt.close('all')

def B6(data, savename):

    specs = [d for d in data.values()]

    fig, axes = plt.subplots(3,1,figsize=(5,8))

    fig.suptitle('Voltage vs Time')

    axes[0].set_title('Liquid Water')
    axes[1].set_title('Partially Frozen Water')
    axes[2].set_title('Fully Frozen Water')

    for i in range(0,3):

        j = closest(0, specs[i]['TIME'].to_numpy())

        axes[i].plot(specs[i]['TIME'].to_numpy()[j+50:-50], smooth(specs[i]['CH1 Peak Detect'].to_numpy(), 30)[j+50:-50], color='black')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Smoothed Voltage (V)')

        axes[i].set_ylim(-10, -5)


    plt.tight_layout()

    savepath = os.path.join('results', 'plots', 'B6_Plot.png')
    plt.savefig(savepath)
    plt.close('all')

""" EXPERIMENTS """

def main():

    B1(load('00','01', '02'), 'b1_water')

    B2(load('03'), 'b2_water')
    B2(load('04'), 'b2_rubber')
    B2(load('05'), 'b2_ethanol')

    B3(load('07', '12'), 'b3_water')
    B3(load('08', '13'), 'b3_rubber')
    B3(load('10', '11'), 'b3_ethanol')

    B4(load('14', '15', '16', '17', '18', '19'), 'b4_water')
    B4(load('20', '22', '23', '24', '25', '26'), 'b4_rubber')
    B4(load('27', '28', '29', '30', '34', '33', '35'), 'b4_ethanol')

    B5(load('36', '37', '38', '39', '40'), 'b5_water')
    B5(load('41', '42', '43', '44', '45'), 'b5_rubber')
    B5(load('46', '47', '48', '49', '50'), 'b5_ethanol')

    B6(load('51', '53', '52'), 'b6_water')

if __name__ == '__main__':
    main()
