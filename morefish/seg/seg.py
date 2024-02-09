from cellpose import models as cp_models
import torch
import geopandas as gpd
import pandas as pd
import cv2
from shapely import geometry
import numpy as np
from loguru import logger
import tqdm
import sys
from ..utils import geometry_colname


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
    poly = poly.simplify(simplify_eps)
    return poly



class Segmentor:
    def __init__(self, mr, name, *, model=None, model_kws={}):
        self.model = model
        if self.model is None:
            self.model = CellPoseModel(**model_kws)
        self._seg_prefices = set()
        self.mr = mr
        self.name = name
        self._prepare_dir()

    def _prepare_dir(self):
        self.reseg_dir = self.mr.reseg_dir/self.name
        self.reseg_dir.mkdir(exist_ok=True, parents=True)
        self.tile_dir = self.reseg_dir/'tiles'
        self.tile_dir.mkdir(exist_ok=True, parents=True)
        self.partition_dir = self.reseg_dir/'partitions'
        self.partition_dir.mkdir(exist_ok=True, parents=True)


    @classmethod
    def _formart_seg_prefix(cls, tile_i, z):
        return f'{str(tile_i).zfill(6)}{str(z).zfill(2)}'
    

    ##TODO may be change "all" to more flexible options
    def segment_all_tiles(self, stains, z, override=False,
                          diameter=None, 
                          channels=[1,2], gpu=True, simplify_eps=0.5,):

        logger.info(f'Segmenting each tile for task "{self.name}". '
                    f'{"Overriding existing segmentations." if override else "Skipping existing segmentations."}')
        logger.remove()
        logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)

        outfn_prefix = '+'.join([stain for stain in stains if stain is not None])+f'_z{z}'
        gdfs = []
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for tile_i,tile_bbox, in enumerate(tqdm.tqdm(self.mr.tiles.tiles)):
                outfn = self.tile_dir/f'{outfn_prefix}_{tile_i}.parquet'

                if outfn.exists() and not override:
                    gdf = gpd.read_parquet(outfn)
                else:
                    gdf = self.model.segment(self.mr.get_tile_image(stains, z,tile_i), diameter=diameter, channels=channels, 
                                             ##TODO this xy seems wrong
                                            origin_yx=tile_bbox[:2], 
                                            img_prefix=self._formart_seg_prefix(tile_i, z), 
                                            # img_prefix=self._gen_img_prefix(), z_prefix=z, 
                                            simplify_eps=simplify_eps)
                    gdf['geometry'] = gdf['geometry'].apply(self.mr.unit_transform.mosaic_to_micron)
                    gdf['ZIndex'] = z
                    gdf.to_parquet(outfn)
                gdfs.append(gdf)

        logger.remove()
        logger.add(sys.stderr)

        return gdfs

    
    @classmethod
    def find_ovlp_btwn_segs(cls, gdf1, gdf2):
        ##TODO [contains, within, overlaps]
        ##TODO this has some problem with 'how="inner"'
        # return gpd.sjoin(gdf1, gdf2, how="inner", predicate="overlaps")
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
    # def _subset_segs(cls, seg_gdf, mr, bbox=None, tile_i=None):

    #     if bbox is None and tile_i is None:
    #         raise ValueError('Either bbox or tile_i must be specified')
    #     elif bbox is not None and tile_i is not None:
    #         raise ValueError('Only one of bbox or tile_i can be specified')
    #     elif tile_i is not None:
    #         bbox = mr.tiles.tiles[tile_i]
        
    @classmethod
    # def _partition_transcripts(cls, seg_gdf, transcript_coords):
    def _partition_transcripts(cls, seg_gdf, transcript_df):
        geocol = geometry_colname(seg_gdf)
        part = []
        # for x,y,*_ in transcript_coords:
        for _,(x,y,*_) in transcript_df.iterrows():
            cands = seg_gdf.sindex.intersection([x,y,x,y])
            point = geometry.Point(x,y)

            good = []
            for ci in cands:
                if point.within(seg_gdf.iloc[ci][geocol]) or point.touches(seg_gdf.iloc[ci][geocol]):
                    good.append(ci)

            if len(good)==0:
                part.append(-1)
            elif len(good)>1:
                logger.debug(f'Multiple candidates for {x,y}: {good}')
                part.append(-1)
                # raise
            else:
                part.append(good[0])
        return part


    def partition_transcripts(self, seg_gdf, override=False, chunk_size=100000):
        ## TODO parallelize this function

        logger.info(f'Partitioning transcripts into cells for task "{self.name}". '
                    f'Overriding existing partitions' if override else 'Skipping existing partitions')
        logger.remove()
        logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)

        
        rlt = []
        # for i in tqdm.tqdm(range(0, len(self.mr.transcripts.coords_micron), chunk_size)):
        #     outfn = self.partition_dir/f'{i}-{min(i+chunk_size, len(self.mr.transcripts.coords_micron))}.parts.npy'
        for i in tqdm.tqdm(range(0, len(self.mr.transcripts.df), chunk_size)):
            outfn = self.partition_dir/f'{i}-{min(i+chunk_size, len(self.mr.transcripts.df))}.parts.npy'
            if outfn.exists() and not override:
                part = np.load(outfn).tolist()
            else:      
                # chunk = self.mr.transcripts.coords_micron[i:i+chunk_size]
                chunk = self.mr.transcripts.df.iloc[i:i+chunk_size]
                part = self._partition_transcripts(seg_gdf, chunk)
                np.save(outfn, part)
            rlt.extend(part)

        logger.remove()
        logger.add(sys.stderr)
        df = pd.DataFrame(np.vstack((rlt,self.mr.transcripts.df['gene'])).T, columns=['cell','gene'])
        # df = pd.DataFrame(np.vstack((rlt,self.mr.transcripts.genes)).T, columns=['cell','gene'])
        cxg = df[df['cell']!=-1].pivot_table(index='cell', columns='gene', 
                                            aggfunc='size', fill_value=0)
        cxg.index = seg_gdf.index[cxg.index]
        cxg = cxg.reindex(seg_gdf.index).fillna(0).astype(int)
        cxg.index.name='cell'
        return cxg        


    
# def partition_transcripts(seg_gdf, mr, chunk_size=100000):
#     ## TODO parallelize this function
#     logger.info('Partitioning transcripts into cells')
#     logger.remove()
#     logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)
#     geocol = geometry_colname(seg_gdf)

#     rlt = []
#     for x,y,*_ in tqdm.tqdm(mr.transcripts.coords_micron):
#         cands = seg_gdf.sindex.intersection([x,y,x,y])
#         point = geometry.Point(x,y)

#         good = []
#         for ci in cands:
#             # if point.within(seg_gdf.iloc[ci].geometry) or point.touches(seg_gdf.iloc[ci].geometry):
#             if point.within(seg_gdf.iloc[ci][geocol]) or point.touches(seg_gdf.iloc[ci][geocol]):
#                 good.append(ci)
#         if len(good)==0:
#             rlt.append(-1)
#         elif len(good)>1:
#             logger.debug(f'Multiple candidates for {x,y}: {good}')
#             rlt.append(-1)
#             # raise
#         else:
#             rlt.append(good[0])
#     logger.remove()
#     logger.add(sys.stderr)

#     logger.info('Compiling cell x gene matrix')

#     df = pd.DataFrame(np.vstack((rlt,mr.transcripts.genes)).T, columns=['cell','gene'])
#     return df

# def get_cell_x_gene_matrix(seg_gdf, mr):
#     df = partition_transcripts(seg_gdf, mr)
#     cxg = df[df['cell']!=-1].pivot_table(index='cell', columns='gene', 
#                                         aggfunc='size', fill_value=0)
#     cxg.index = seg_gdf.index[cxg.index]
#     cxg = cxg.reindex(seg_gdf.index).fillna(0).astype(int)
#     cxg.index.name='cell'
#     return cxg