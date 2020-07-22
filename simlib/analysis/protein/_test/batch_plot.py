
import pandas as pd
import uplot as u

df = pd.read_csv('batch.csv')
df['residue'] += 1

batches = df['batch'].unique()
codes = [column for column in df.columns if column not in ['batch', 'residue']]

figure = u.figure(style={
    'x_title': r'$residue,\ i$',
    'y_title': r'$secondary\ structure$',
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
        label = '{}_{}'.format(code, batch) if batch == 0 else None
        figure += u.line(df_['residue'], df_[code], style={
            'label': label,
#            'label': '{}_{}'.format(code, batch), 
#            'linestyle': linestyles[batch],
            'color': 'C%s' % i
        })

#for figure_object in figure._figure_objects:
#    print(figure_object._data)

fig, ax = figure.to_mpl(show=False)
fig.savefig('batch_plot.svg')
