import geopandas as gpd
import anndata
from ..merfish import MerfishRegion
from shapely import geometry
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from loguru import logger

def get_imshow_extent(bbox, mr):
    xmin,ymin,xmax,ymax = bbox.bounds
    (xmin_,ymin_),(xmax_,ymax_) = mr.unit_transform.mosaic_to_micron(np.array([[xmin-0.5,ymin-0.5],
                                                                               [xmax+0.5,ymax+0.5],]))
    # (xmin_,xmax_),(ymin_,ymax_) = mr.unit_transform.mosaic_to_micron(np.array([[xmin-0.5,ymin+0.5],
    #                                                                            [xmax-0.5,ymax+0.5],]))
    # (xmin_,xmax_),(ymin_,ymax_) = mr.unit_transform.mosaic_to_micron(np.array([[xmin,ymin],
    #                                                                            [xmax,ymax],]))

    return [xmin_,xmax_,ymin_,ymax_]


def imgplot_bbox(ax, bbox, mr, stain, z, **kwargs):
    img = mr.get_stain_image(stain, z, bbox)
    # y,x = img.shape
    extent = get_imshow_extent(bbox, mr)
    # logger.debug(f'px bbox:{bbox}')
    # logger.debug(f'um extent: {extent}')
    ax.imshow(img, extent=extent, 
              origin='lower', 
              aspect='equal',
              **kwargs)
    ax.set_aspect('equal')
    ax.set_xlim(*extent[:2])
    ax.set_ylim(*extent[2:])
    return ax

def imgplot(ax, mr, stain, z, tile_idx, **kwargs):
    bbox = mr.tiles.tiles[tile_idx]
    return imgplot_bbox(ax, bbox, mr, stain, z, **kwargs)












# def relocate(geo, w,h):
#     def relocate_polygon(poly, w,h):
#         xs,ys=poly.exterior.xy
#         return geometry.Polygon(zip(np.array(xs) - w,np.array(ys) - h))

#     if isinstance(geo, geometry.Point):
#         return geometry.Point(geo.x-w, geo.y-h)
#     elif isinstance(geo, geometry.Polygon):
#         return relocate_polygon(geo, w,h )
#     elif isinstance(geo, geometry.MultiPolygon):
#         return geometry.MultiPolygon([relocate_polygon(poly,w,h) for poly in geo.geoms])
#     elif isinstance(geo, geometry.GeometryCollection):
#         return geo
#     else:
#         raise TypeError



def geoplot_bbox(ax, bbox, gdf, style={}):
    geo_styles = {
        'Point': {'color': 'r', 'marker':'.', 'markersize': 1},
        'Polygon': {'facecolor': 'none', 'edgecolor': 'b', 'linewidth': 1},
        'MultiPolygon': {'facecolor': 'none', 'edgecolor': 'b', 'linewidth': 1},
    }
    geo_styles.update(style)

    window = geometry.Polygon([(bbox.x,        bbox.y       ), 
                               (bbox.x+bbox.w, bbox.y       ), 
                               (bbox.x+bbox.w, bbox.y+bbox.h), 
                               (bbox.x,        bbox.y+bbox.h)])
    geo_col = gdf.columns[gdf.dtypes=='geometry'][0]
    geos=gdf[gdf[geo_col].apply(window.intersects)]
    # geos[geo_col] = geos[geo_col].apply(lambda g:relocate(g, bbox.x, bbox.y))

    for geo_type, style in geo_styles.items():
        subset = geos[geos.geometry.type == geo_type]
        subset.plot(ax=ax, **style)
    #geos.plot(ax = ax, facecolor='none', color='none', edgecolor='r', linewidth=1)
    

def boundaryplot_bbox(ax, bbox, mr_or_adata, name=None, adata=None):
    wi,hi,ws,hs = bbox
    if isinstance(mr_or_adata, MerfishRegion):
        ad = mr_or_adata.load_reseg_results(name)
    elif isinstance(mr_or_adata, anndata.AnnData):
        ad = adata
    else:
        raise ValueError()
    window = geometry.Polygon([(wi, hi), (wi+ws, hi), (wi+ws, hi+hs), (wi, hi+hs)])
    geos = ad.uns['cell_boundary_mosaic'][
                    ad.uns['cell_boundary_mosaic']['Geometry'].apply(window.intersects)
            ]['Geometry']
    geos = gpd.GeoDataFrame(geometry=geos, index=geos.index)
   
    geoplot_bbox(ax, bbox, geos)



def boundaryplot_bbox_bk(ax, bbox, mr_or_adata, name=None, adata=None):

    wi,hi,ws,hs = bbox
    if isinstance(mr_or_adata, MerfishRegion):
        ad = mr_or_adata.load_reseg_results(name)
    elif isinstance(mr_or_adata, anndata.AnnData):
        ad = adata
    else:
        raise ValueError()
    window = geometry.Polygon([(wi, hi), (wi+ws, hi), (wi+ws, hi+hs), (wi, hi+hs)])
    geos = ad.uns['cell_boundary_mosaic'][
                    ad.uns['cell_boundary_mosaic']['Geometry'].apply(window.intersects)
            ]['Geometry']
    def relocate(xx):
        ##TODO need to take care of different geometries
        try:
            xs,ys=xx.exterior.xy
            return geometry.Polygon(zip(np.array(xs) - wi,np.array(ys) - hi))
        except:
            return xx
    
    geos = gpd.GeoDataFrame(geometry=geos.apply(relocate), )
    geos.plot(ax = ax, facecolor='none', color='none', edgecolor='r', linewidth=1)

def boundaryplot(ax, tile_idx, mr_or_adata, name=None, adata=None):
    bbox = mr_or_adata.tiles.tiles[tile_idx]
    return boundaryplot_bbox(ax, bbox, mr_or_adata, name, adata)

# def transcriptplot_bbox(ax, bbox, mr):
#     wi,hi,ws,hs = bbox
#     window = geometry.Polygon([(wi, hi), (wi+ws, hi), (wi+ws, hi+hs), (wi, hi+hs)])
#     geos = mr.detected_transcripts[
#                         mr.detected_transcripts['position_mosaic'].apply(window.intersects)
#             ]['position_mosaic']
#     geos = gpd.GeoDataFrame(geometry=geos)
   
#     geoplot_bbox(ax, bbox, geos)

# def transcriptplot(ax, tile_idx, mr):
#     bbox = mr.tiles.tiles[tile_idx]
#     return transcriptplot_bbox(ax, bbox, mr)



# def tileplot(mr, ax=None):
#     ## this works with imshow
#     if ax is None:
#         fig, ax = plt.subplots()
#     for i, (x, y, w, h) in enumerate(mr.tiles.tiles):
#         color = (1,1,1)  # white
#         alpha = 0.1
#         rectangle = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='k', facecolor=color + (alpha,))
#         ax.add_patch(rectangle)
#         ax.text(x+w/2, y+h/2, str(i), ha='center', va='center',)
 
