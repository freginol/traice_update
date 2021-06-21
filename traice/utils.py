def dump_dataframe(df, num_rows=2, label=''):
    if len(label) > 0:
        print(label + ':')
    print('shape:', df.shape)
    print('first', num_rows, 'rows:')
    print(df[0:num_rows])
    print('last', num_rows, 'rows:')
    print(df[df.shape[0]-num_rows:df.shape[0]])
