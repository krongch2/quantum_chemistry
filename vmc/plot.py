import pandas as pd
import matplotlib.pyplot as plt

raw = pd.read_csv('data/collect.csv')

def average_configs(df):
    df['total'] = df['v_ee'] + df['v_en'] + df['ke']
    df['nonint'] = df['v_en'] + df['ke']
    res = pd.Series({
        'v_ee': df['v_ee'].mean(),
        'v_en': df['v_en'].mean(),
        'ke': df['ke'].mean(),
        'nonint': df['nonint'].mean(),
        'total': df['total'].mean(),
        'v_ee_err': df['v_ee'].std()/len(df['v_ee'])**0.5,
        'v_en_err': df['v_en'].std()/len(df['v_en'])**0.5,
        'ke_err': df['ke'].std()/len(df['ke'])**0.5,
        'nonint_err': df['nonint'].std()/len(df['nonint'])**0.5,
        'total_err': df['total'].std()/len(df['total'])**0.5
    })
    res['v'] = res['v_en'] + res['v_ee']
    res['v_err'] = (res['v_ee_err']**2 + res['v_en_err']**2)**0.5
    return res

d = raw.groupby(['alpha', 'beta']).apply(average_configs).reset_index()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
for beta in d.beta.unique():
    dd = d.loc[d.beta == beta, :]
    ax = axs[0, 0]
    ax.errorbar(dd.alpha, dd.nonint, yerr=dd.nonint_err, fmt='o-', label=beta)
    ax.set_ylabel('nonint')

    ax = axs[0, 1]
    ax.errorbar(dd.alpha, dd.total, yerr=dd.total_err, fmt='o-', label=beta)
    ax.set_ylabel('total')

    ax = axs[1, 0]
    ax.errorbar(dd.alpha, dd.ke, yerr=dd.ke_err, fmt='o-', label=beta)
    ax.set_ylabel('kinetic')
    ax.set_xlabel('alpha')

    ax = axs[1, 1]
    ax.errorbar(dd.alpha, dd.v, yerr=dd.v_err, fmt='o-', label=beta)
    ax.set_ylabel('potential')
    ax.set_xlabel('alpha')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.tight_layout()
plt.show()
