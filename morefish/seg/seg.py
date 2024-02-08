from cellpose import models as cp_models
#from cellpose import io as cp_io
import torch
import geopandas as gpd
import pandas as pd
import cv2
from shapely import geometry
import numpy as np
from loguru import logger
import tqdm
import sys



class CellPoseModel:
    def __init__(self, *, model_type=None, pretrained_path=None, gpu=True):
        if pretrained_path is None and model_type is None:
            raise ValueError("Either pretrained_path or model_type must be specified")
        if pretrained_path is not None and model_type is not None:
            raise ValueError("Only one of pretrained_path or model_type can be specified")
        if pretrained_path is not None:
            self._model = cp_models.CellposeModel(pretrained_model=pretrained_path, 
                                                  gpu=gpu&torch.cuda.is_available())
        else:
            self._model = cp_models.CellposeModel(model_type=model_type, 
                                                  gpu=gpu&torch.cuda.is_available())

    def segment(self, img, diameter=None, channels=[1,2], 
                  channel_axis=2, origin_yx=[0,0], img_prefix='000000_000', 
                  simplify_eps=0.5):
        # masks, flows, styles = self._model.eval(img, diameter=diameter, 
        masks, *_ = self._model.eval(img, diameter=diameter, 
                                     channels=channels,
                                     channel_axis=channel_axis)
        geos = []
        seg_ids = []
        for mask_i in range(1, masks.max()):
            seg_ids.append(f'{img_prefix}_{str(mask_i).zfill(6)}')
            geos.append( polygons_from_masks(masks, mask_i, origin_yx, simplify_eps) )
        gdf = gpd.GeoDataFrame(index=seg_ids, geometry=geos)
        return gdf
    
def polygons_from_masks(masks, mask_id, mask_origin_yx = [0,0], simplify_eps = 0.5):
    contours,_ = cv2.findContours((masks==mask_id).astype('uint8'), 
                                    cv2.RETR_LIST, 
                                    cv2.CHAIN_APPROX_SIMPLE,)

    ## having len(c)>=3 is important to remove bad contours, like two pixels 
    poly = geometry.MultiPolygon(geometry.Polygon(c.squeeze()+mask_origin_yx) for c in contours if len(c)>=3) 
#         poly = geometry.MultiPolygon(geometry.Polygon(c.squeeze()+mask_origin_yx[::-1]) for c in contours)
    poly = poly.simplify(simplify_eps)
    return poly


def geocol(gdf):
    return gdf.columns[gdf.dtypes=='geometry']

class Segmentor:
    def __init__(self, *, model=None, model_kws={}):
        self.model = model
        if self.model is None:
            self.model = CellPoseModel(**model_kws)
        self._seg_prefices = set()

    # def _gen_img_prefix(self):
    #     prefix = np.random.randint(100000)
    #     while prefix in self._seg_prefices:
    #         prefix = np.random.randint(100000)
    #     self._seg_prefices.update([prefix])
    #     return prefix
    
    @classmethod
    def formart_seg_prefix(cls, tile_i, z):
        return f'{str(tile_i).zfill(6)}{str(z).zfill(2)}'
    
    def segment_all_tiles(self, mr, stains, z, name, override=False,
                        #   ovlp_merge_ratio=0.5, 
                          diameter=None, 
                          channels=[1,2], gpu=True, simplify_eps=0.5):
        reseg_dir = mr.reseg_dir/name
        reseg_dir.mkdir(exist_ok=True, parents=True)
        tile_dir = reseg_dir/'tiles'
        tile_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f'Segmenting each tile for task "{name}". '
                    f'{"Overriding existing segmentations." if override else "Skipping existing segmentations."}')
        logger.remove()
        logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)

        outfn_prefix = '+'.join([stain for stain in stains if stain is not None])+f'_z{z}'
        gdfs = []
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for tile_i,tile_bbox, in enumerate(tqdm.tqdm(mr.tiles.tiles)):
                outfn = tile_dir/f'{outfn_prefix}_{tile_i}.parquet'

                if outfn.exists() and not override:
                    gdf = gpd.read_parquet(outfn)
                else:
                    gdf = self.model.segment(mr.get_tile_image(stains, z,tile_i), diameter=diameter, channels=channels, 
                                            origin_yx=tile_bbox[:2], 
                                            img_prefix=self.formart_seg_prefix(tile_i, z), 
                                            # img_prefix=self._gen_img_prefix(), z_prefix=z, 
                                            simplify_eps=simplify_eps)
                    gdf['geometry'] = gdf['geometry'].apply(mr.unit_transform.mosaic_to_micron)
                    gdf['ZIndex'] = z
                    gdf.to_parquet(outfn)
                gdfs.append(gdf)

        logger.remove()
        logger.add(sys.stderr)
        # logger.info(f'Tile segmentation completed')

        return gdfs

    
        # logger.info('Initializing cellpose model')
        # cpsegor = CellPoseSegmentor(model_type=model_type, pretrained_path=pretrained_path, gpu=gpu)
        # tile_segs = []


        # logger.info('Segmenting each tile')
        # logger.remove()
        # logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)

        # import warnings
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=UserWarning)
        #     for tile_i,tile_info, in enumerate(tqdm.tqdm(mr.tiles.tiles)):
        #         gdf = cpsegor.seg_image(mr.get_tile_image(tile_i, stains, z), 
        #                                 diameter=diameter, channels=channels, 
        #                                 origin_yx=tile_info[:2])
        #         tile_segs.append(gdf)

        # logger.remove()
        # logger.add(sys.stderr)
            

        # logger.info('Segmenting tile segmentations')
        # segs = cpsegor.compile_segs(tile_segs, ovlp_merge_ratio)
        # segs['geometry'] = segs['geometry'].apply(mr.unit_transform.mosaic_to_micron)
        # segs.index = cell_prefix+segs.index
        # return segs

    # def _init_model(self, *, model_type=None, pretrained_path=None, gpu=True):
    #     if pretrained_path is None and model_type is None:
    #         raise ValueError("Either pretrained_path or model_type must be specified")
    #     if pretrained_path is not None and model_type is not None:
    #         raise ValueError("Only one of pretrained_path or model_type can be specified")
    #     if pretrained_path is not None:
    #         self._model = cp_models.CellposeModel(pretrained_model=pretrained_path, 
    #                                              gpu=gpu&torch.cuda.is_available())
    #     else:
    #         self._model = cp_models.CellposeModel(model_type=model_type, 
    #                                              gpu=gpu&torch.cuda.is_available())

    # @classmethod
    # def _polygons_from_masks(cls, masks, mask_id, mask_origin_yx = [0,0], simplify_eps = 0.5):
    #     contours,_ = cv2.findContours((masks==mask_id).astype('uint8'), 
    #                                   cv2.RETR_LIST, 
    #                                   cv2.CHAIN_APPROX_SIMPLE,)
    #     poly = geometry.MultiPolygon(geometry.Polygon(c.squeeze()+mask_origin_yx) for c in contours)
    #     # poly = geometry.MultiPolygon(geometry.Polygon(c.squeeze()+mask_origin_yx[::-1]) for c in contours)
    #     poly = poly.simplify(simplify_eps)
    #     return poly
        
    # def seg_image(self, img, diameter=None, channels=[1,2], 
    #               channel_axis=2, origin_yx=[0,0]):
    #     masks, *_ = self._model.eval(img, diameter=diameter, 
    #                                  channels=channels,
    #                                  channel_axis=channel_axis)
    #     prefix = self._gen_seg_prefix()
    #     geos = []
    #     seg_ids = []
    #     for mask_i in range(1, masks.max()):
    #         seg_ids.append(f'{str(prefix).zfill(6)}{str(mask_i).zfill(6)}')
    #         geos.append( self._polygons_from_masks(masks, mask_i, origin_yx) )
    #     gdf = gpd.GeoDataFrame(index=seg_ids, geometry=geos)
    #     return gdf
    
    @classmethod
    def find_ovlp_btwn_segs(cls, gdf1, gdf2):
        return gpd.sjoin(gdf1, gdf2, how="inner", predicate="intersects")

    @classmethod
    def resolve_ovlp_btwn_segs(cls, gdf1, gdf2, merge_ratio=0.5):
        ##TODO currently designed to resolve overlaps of the same z-index. not considering overlaps across z-indices yet.
        
        ovlpgdf = cls.find_ovlp_btwn_segs(gdf1, gdf2)
        
        to_remove = set()
        to_remove.update(ovlpgdf[ovlpgdf.index.duplicated()].index.tolist())
        to_remove.update(ovlpgdf[ovlpgdf['index_right'].duplicated()]['index_right'].tolist())
        
        
        ovlpgdf = ovlpgdf[~(ovlpgdf.index.isin(to_remove)|ovlpgdf['index_right'].isin(to_remove))].copy()
        # _gdf1 = _gdf1[~_gdf1.index.isin(to_remove)]
        # _gdf2 = _gdf2[~_gdf2.index.isin(to_remove)]
        _gdf1 = gdf1[~gdf1.index.isin(to_remove)]
        _gdf2 = gdf2[~gdf2.index.isin(to_remove)]


        ovlpgdf['area_ovlp'] = ovlpgdf['geometry'].area
        ovlpgdf['ratio1'] = ovlpgdf['area_ovlp']/_gdf1.loc[ovlpgdf.index]['geometry'].area.values
        ovlpgdf['ratio2'] = ovlpgdf['area_ovlp']/_gdf2.loc[ovlpgdf['index_right']]['geometry'].area.values

        solved = []
        ids = []
        for id1,row in ovlpgdf.iterrows():
            # *_, id2, _, r1, r2 = row
            id2, r1, r2 = row[['index_right', 'ratio1', 'ratio2']]

            if max(r1, r2)>=merge_ratio:
                ##TODO handle a warning raising in union
                poly = _gdf1.loc[id1, 'geometry'].union(_gdf2.loc[id2, 'geometry'])                
            elif r2 > r1: # area1 > area2
                poly = _gdf1.loc[id1, 'geometry'].difference(_gdf2.loc[id2, 'geometry'])
            else:
                poly = _gdf2.loc[id2, 'geometry'].difference(_gdf1.loc[id1, 'geometry'])
            solved.append(poly)

            ids.append(id1 if r2 > r1 else id2)

        solved = pd.concat([_gdf1[~_gdf1.index.isin(ovlpgdf.index)],
                            _gdf2[~_gdf2.index.isin(ovlpgdf['index_right'])],
                            gpd.GeoDataFrame(index=ids, geometry=solved)])
#         solved = solved[~solved.index.isin(to_remove)]
        
        return solved        


    @classmethod
    def _compile_segs(cls, segs, merge_ratio=0.5):
        compiled = None
        for seg in segs:
            if compiled is None:
                compiled = seg
            else:
                compiled = cls.resolve_ovlp_btwn_segs(compiled, seg, merge_ratio)
        return compiled
    
    @classmethod
    def compile_segs(cls, segs, *, merge_ratio=0.5, chunk_size=20, n_cpus=1,):
        if n_cpus>1:
            import multiprocessing
            import itertools
            logger.warning('"n_cpus=1" is recommended for segmentation compilation. Multiprocessing may not provide significant speedup.')

            pool = multiprocessing.Pool(n_cpus)
        try:
            while len(segs) > 1:
                segs = [segs[i:i+chunk_size] for i in range(0, len(segs), chunk_size)]
                if n_cpus > 1:
                    segs = pool.starmap(cls._compile_segs, itertools.product(segs, [merge_ratio]))
                else:
                    segs = [cls._compile_segs(x, merge_ratio=merge_ratio) for x in segs]

        except Exception as e:
            print("Error occurred:", e)

        finally:
            if n_cpus > 1:
                pool.close()
                pool.join()
            
        segs, = segs
        return segs

    # @classmethod
    # def partition_transcripts(cls, seg_gdf, mr):
    #     pt_gdf = gpd.GeoDataFrame([geometry.Point(x,y) for x,y in mr.detected_transcripts[['global_x','global_y']].values])
    #     within = gpd.sjoin(pt_gdf, seg_gdf, how='left', op='within')

    #     # within = partition_points(seg_gdf, 
    #     #                           mr.detected_transcripts[['global_x','global_y']].values)
        
    #     within = within.loc[within.index.drop_duplicates(keep=False)]
    #     within = within.rename(columns={'index_right':'cell'})    
    #     within['gene'] = mr.detected_transcripts.iloc[within.index]['gene'].values
        
    #     cxg = within.dropna().groupby(['cell','gene'])['gene'].count().to_frame('num')
    #     cxg = cxg.reset_index().pivot(index = 'cell', columns='gene', values='num')
    #     cxg = cxg.reindex(seg_gdf.index).fillna(0).astype(int)
    #     cxg.columns.name=None
    #     cxg.index.name='cell'
        
    #     def sort_cxg_cols(_cxg):
    # #         blank_cols = _cxg.columns.str.startswith('Blank-')
    # #         _cxg = _cxg[_cxg.columns[~blank_cols].tolist()+_cxg.columns[blank_cols].tolist()]
    #         return _cxg
    #     cxg = sort_cxg_cols(cxg)
    #     return cxg
    
    @classmethod
    def _subset_segs(cls, seg_gdf, mr, bbox=None, tile_i=None):

        if bbox is None and tile_i is None:
            raise ValueError('Either bbox or tile_i must be specified')
        elif bbox is not None and tile_i is not None:
            raise ValueError('Only one of bbox or tile_i can be specified')
        elif tile_i is not None:
            bbox = mr.tiles.tiles[tile_i]
        


# def segmentation_compile_image(mr, stains, z, *, model_type=None, pretrained_path=None, cell_prefix = 'cell_',
#                                gpu=True, diameter=None, 
#                                channels=[1,2], ovlp_merge_ratio=0.5):
#     logger.info('Initializing cellpose model')
#     cpsegor = CellPoseSegmentor(model_type=model_type, pretrained_path=pretrained_path, gpu=gpu)
#     tile_segs = []


#     logger.info('Segmenting each tile')
#     logger.remove()
#     logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)

#     import warnings
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=UserWarning)
#         for tile_i,tile_info, in enumerate(tqdm.tqdm(mr.tiles.tiles)):
#             gdf = cpsegor.seg_image(mr.get_tile_image(tile_i, stains, z), 
#                                     diameter=diameter, channels=channels, 
#                                     origin_yx=tile_info[:2])
#             tile_segs.append(gdf)

#     logger.remove()
#     logger.add(sys.stderr)
        

#     logger.info('Segmenting tile segmentations')
#     segs = cpsegor.compile_segs(tile_segs, ovlp_merge_ratio)
#     segs['geometry'] = segs['geometry'].apply(mr.unit_transform.mosaic_to_micron)
#     segs.index = cell_prefix+segs.index
#     return segs

# def partition_points(seg_gdf, point_xy):
#     pt_gdf = gpd.GeoDataFrame(geometry=[geometry.Point(x,y) for x,y in point_xy])
#     within = gpd.sjoin(pt_gdf, seg_gdf, how='left', op='within')
#     return within

# def partition_transcripts(seg_gdf, mr):
#     pt_gdf = gpd.GeoDataFrame([geometry.Point(x,y) for x,y in mr.detected_transcripts[['global_x','global_y']].values])
#     within = gpd.sjoin(pt_gdf, seg_gdf, how='left', op='within')

#     # within = partition_points(seg_gdf, 
#     #                           mr.detected_transcripts[['global_x','global_y']].values)

#     within = within.loc[within.index.drop_duplicates(keep=False)]
#     within = within.rename(columns={'index_right':'cell'})    
#     within['gene'] = mr.detected_transcripts.iloc[within.index]['gene'].values
    
#     cxg = within.dropna().groupby(['cell','gene'])['gene'].count().to_frame('num')
#     cxg = cxg.reset_index().pivot(index = 'cell', columns='gene', values='num')
#     cxg = cxg.reindex(seg_gdf.index).fillna(0).astype(int)
#     cxg.columns.name=None
#     cxg.index.name='cell'
    
#     def sort_cxg_cols(_cxg):
# #         blank_cols = _cxg.columns.str.startswith('Blank-')
# #         _cxg = _cxg[_cxg.columns[~blank_cols].tolist()+_cxg.columns[blank_cols].tolist()]
#         return _cxg
#     cxg = sort_cxg_cols(cxg)
#     return cxg
    
def partition_transcripts(seg_gdf, mr):
    ## TODO parallelize this function
    logger.info('Partitioning transcripts into cells')
    logger.remove()
    logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)
    geo_col = seg_gdf.columns[seg_gdf.dtypes=='geometry'][0]

    rlt = []
    for x,y,*_ in tqdm.tqdm(mr.transcripts.coords_micron):
        cands = seg_gdf.sindex.intersection([x,y,x,y])
        point = geometry.Point(x,y)

        good = []
        for ci in cands:
            # if point.within(seg_gdf.iloc[ci].geometry) or point.touches(seg_gdf.iloc[ci].geometry):
            if point.within(seg_gdf.iloc[ci][geo_col]) or point.touches(seg_gdf.iloc[ci][geo_col]):
                good.append(ci)
        if len(good)==0:
            rlt.append(-1)
        elif len(good)>1:
            raise
        else:
            rlt.append(good[0])
    logger.remove()
    logger.add(sys.stderr)

    logger.info('Compiling cell x gene matrix')

    df = pd.DataFrame(np.vstack((rlt,mr.transcripts.genes)).T, columns=['cell','gene'])
    return df

def get_cell_x_gene_matrix(seg_gdf, mr):
    df = partition_transcripts(seg_gdf, mr)
    cxg = df[df['cell']!=-1].pivot_table(index='cell', columns='gene', 
                                        aggfunc='size', fill_value=0)
    cxg.index = seg_gdf.index[cxg.index]
    cxg = cxg.reindex(seg_gdf.index).fillna(0).astype(int)
    cxg.index.name='cell'
    return cxg