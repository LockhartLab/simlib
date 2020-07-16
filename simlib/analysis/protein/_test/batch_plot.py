
import pandas as pd
import uplot as u

df = pd.read_csv('batch.csv')

batches = df['batch'].unique()
codes = [column for column in df.columns if column not in ['batch', 'residue']]

figure = u.figure(style={
    'xtitle': r'$residue,\ i$',
    'ytitle': r'$secondary\ structure$',
    'legend': True
})

linestyles = {
    0: 'solid',
    1: 'dashed'
}

for batch in batches:
    df_ = df.query("batch == %s" % batch)
    print(df_)
    for i, code in enumerate(codes):
        figure += u.line(df_['residue'], df_[code], style={
            'label': '{}_{}'.format(code, batch), 
            'linestyle': linestyles[batch],
            'color': 'C%s' % i
        })

#for figure_object in figure._figure_objects:
#    print(figure_object._data)

figure.to_mpl(show=False).savefig('batch_plot.svg')
