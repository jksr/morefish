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

class CellPoseSegmentor:
    def __init__(self, *, model_type=None, pretrained_path=None, gpu=True):
        self._init_model(model_type=model_type, pretrained_path=pretrained_path, gpu=gpu)
        self._seg_prefices = set()
#         self.transform = UnitTransform(self.merfish_region.micron_to_mosaic_matrix_path)

    def _gen_seg_prefix(self):
        prefix = np.random.randint(100000)
        while prefix in self._seg_prefices:
            prefix = np.random.randint(100000)
        self._seg_prefices.update([prefix])
        return prefix
    
    def _init_model(self, *, model_type=None, pretrained_path=None, gpu=True):
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

        
    def seg_image(self, img, diameter=None, channels=[1,2], 
                  channel_axis=2, origin_offset=[0,0]):
        masks, flows, styles = self._model.eval(img, diameter=diameter, 
                                               channels=channels, 
                                               channel_axis=channel_axis)
        prefix = self._gen_seg_prefix()
        geos = []
        seg_ids = []
        for mask_i in range(1, masks.max()):
            seg_ids.append(f'{str(prefix).zfill(6)}{str(mask_i).zfill(6)}')
            geos.append( self._polygons_from_masks(masks, mask_i, origin_offset) )
        gdf = gpd.GeoDataFrame(index=seg_ids, geometry=geos)
        return gdf
    
    @classmethod
    def find_ovlp_btwn_segs(cls, gdf1, gdf2):
        return gpd.sjoin(gdf1, gdf2, how="inner", predicate="intersects")

    @classmethod
    def resolve_ovlp_btwn_segs(cls, gdf1, gdf2, merge_ratio=0.5):
        _gdf1 = gdf1.copy()
        _gdf2 = gdf2.copy()
        
        ovlpgdf = cls.find_ovlp_btwn_segs(_gdf1, _gdf2)
        
        to_remove = set()
        to_remove.update(ovlpgdf[ovlpgdf.index.duplicated()].index.tolist())
        to_remove.update(ovlpgdf[ovlpgdf['index_right'].duplicated()]['index_right'].tolist())
        
        
        ovlpgdf = ovlpgdf[~(ovlpgdf.index.isin(to_remove)|ovlpgdf['index_right'].isin(to_remove))].copy()
        _gdf1 = _gdf1[~_gdf1.index.isin(to_remove)]
        _gdf2 = _gdf2[~_gdf2.index.isin(to_remove)]
        
        ovlpgdf['area_ovlp'] = ovlpgdf['geometry'].area
        ovlpgdf['ratio1'] = ovlpgdf['area_ovlp']/_gdf1.loc[ovlpgdf.index]['geometry'].area.values
        ovlpgdf['ratio2'] = ovlpgdf['area_ovlp']/_gdf2.loc[ovlpgdf['index_right']]['geometry'].area.values

        solved = []
        ids = []
        for id1,row in ovlpgdf.iterrows():
            *_, id2, _, r1, r2 = row

            if max(r1, r2)>=merge_ratio:
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
    def compile_segs(cls, segs, merge_ratio=0.5):
        compiled = None
        for seg in segs:
            if compiled is None:
                compiled = seg
            else:
                compiled = cls.resolve_ovlp_btwn_segs(compiled, seg, merge_ratio)
        return compiled
    

    @classmethod
    def _polygons_from_masks(cls, masks, mask_id, mask_origin_yx = [0,0], simplify_eps = 0.5):
        contours,_ = cv2.findContours((masks==mask_id).astype('uint8'), 
                                      cv2.RETR_LIST, 
                                      cv2.CHAIN_APPROX_SIMPLE,)
        poly = geometry.MultiPolygon(geometry.Polygon(c.squeeze()+mask_origin_yx) for c in contours)
#         poly = geometry.MultiPolygon(geometry.Polygon(c.squeeze()+mask_origin_yx[::-1]) for c in contours)
        poly = poly.simplify(simplify_eps)
        return poly

def segmentation_compile_image(mr, stains, z, *, model_type=None, pretrained_path=None, cell_prefix = 'cell_',
                               gpu=True, diameter=None, 
                               channels=[1,2], ovlp_merge_ratio=0.5):
    logger.info('Initializing cellpose model')
    cpsegor = CellPoseSegmentor(model_type=model_type, pretrained_path=pretrained_path, gpu=gpu)
    tile_segs = []


    logger.info('Segmenting each tile')
    logger.remove()
    logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for tile_i,tile_info, in enumerate(tqdm.tqdm(mr.tiles.tiles)):
            gdf = cpsegor.seg_image(mr.read_tile(tile_i, stains, z), 
                                    diameter=diameter, channels=channels, 
                                    origin_offset=tile_info[:2])
            tile_segs.append(gdf)

    logger.remove()
    logger.add(sys.stderr)
        

    logger.info('Segmenting tile segmentations')
    segs = cpsegor.compile_segs(tile_segs, ovlp_merge_ratio)
    segs['geometry'] = segs['geometry'].apply(mr.unit_transform.mosaic_to_micron)
    segs.index = cell_prefix+segs.index
    return segs

def partition_points(seg_gdf, point_xy):
    pt_gdf = gpd.GeoDataFrame(geometry=[geometry.Point(x,y) for x,y in point_xy])
    within = gpd.sjoin(pt_gdf, seg_gdf, how='left', op='within')
    return within

def partition_transcripts(seg_gdf, mr):
    within = partition_points(seg_gdf, 
                              mr.detected_transcripts[['global_x','global_y']].values)
    within = within.loc[within.index.drop_duplicates(keep=False)]
    within = within.rename(columns={'index_right':'cell'})    
    within['gene'] = mr.detected_transcripts.iloc[within.index]['gene'].values
    
    cxg = within.dropna().groupby(['cell','gene'])['gene'].count().to_frame('num')
    cxg = cxg.reset_index().pivot(index = 'cell', columns='gene', values='num')
    cxg = cxg.reindex(seg_gdf.index).fillna(0).astype(int)
    cxg.columns.name=None
    cxg.index.name='cell'
    
    def sort_cxg_cols(_cxg):
#         blank_cols = _cxg.columns.str.startswith('Blank-')
#         _cxg = _cxg[_cxg.columns[~blank_cols].tolist()+_cxg.columns[blank_cols].tolist()]
        return _cxg
    cxg = sort_cxg_cols(cxg)
    return cxg
    
