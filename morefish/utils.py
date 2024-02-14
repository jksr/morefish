def geometry_colname(geo_df):
    sel = geo_df.dtypes=='geometry'
    if sum(sel)>1:
        raise ValueError('More than one geometry column found in the dataframe')
    return geo_df.columns[sel][0]