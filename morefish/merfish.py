import numpy as np
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
import tifffile
import itertools
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import anndata
from shapely import geometry
from shapely.affinity import affine_transform
from loguru import logger

class BBox(tuple):
    def __new__(cls, x, y=None, w=None, h=None):
        if isinstance(x, (list, tuple)):
            return super(BBox, cls).__new__(cls, x)
        elif isinstance(x, (dict, BBox)):
            return super(BBox, cls).__new__(cls, x.x, x.y, x.w, x.h)
        elif all(isinstance(coord, int) for coord in (x, y, w, h)):
            return super(BBox, cls).__new__(cls, (x, y, w, h))
        else:
            raise ValueError('Invalid input for BBox')
    # def __repr__(self):
    #     return f'BBox(x={self[0]}, y={self[1]}, w={self[2]}, h={self[3]})'
    @property
    def x(self):
        return self[0]
    @property 
    def y(self):
        return self[1]
    @property
    def w(self):
        return self[2]
    @property
    def h(self):
        return self[3]
    @property
    def bounds(self):
        return (self.x, self.y, self.x+self.w, self.y+self.h)


class Tiles:
    def __init__(self, w_px, h_px, non_ovlp_px = 2048, 
                 ovlp_px=None, ovlp_ratio=0.1):
        self.w_px = w_px
        self.h_px = h_px
        self.non_ovlap_px = non_ovlp_px
        
        if ovlp_px is None:
            ovlp_px = int(non_ovlp_px * 0.1)
        self.ovlp_px = ovlp_px
        
        self.tiles, self.n_hor, self.n_vert = self.make_tiles()
        
    def make_tiles(self):
        window_px = self.non_ovlap_px + 2 * self.ovlp_px
        step_px = self.non_ovlap_px + self.ovlp_px
        
        tiles = [
                  BBox([x, y,
                   min(self.w_px-x, window_px),
                   min(self.h_px-y, window_px),
                  ]) \
                  for x,y in itertools.product(range(0, self.w_px, step_px), 
                                                 range(0, self.h_px, step_px))
        ]
        n_hor = len(range(0, self.w_px, step_px))
        n_vert = len(range(0, self.h_px, step_px))
        return tiles, n_hor, n_vert
    
    def write_tiles_file(self, fn):
        with open(fn, 'w') as f:
            json.dump({'width':self.w_px,
                       'height':self.h_px,
                       'num_horizontal_tiles':self.n_hor,
                       'num_vertical_tiles':self.n_vert,
                       'non_overlap_size':self.non_ovlap_px,
                       'overelap_size':self.ovlp_px,
                       'tiles':self.tiles,
                      }, f)
    
    def read_tiles_file(self, fn):
        with open(fn) as f:
            j = json.load(f)
        self.w_px = j['width']
        self.h_px = j['height']
        self.n_hor = j['num_horizontal_tiles']
        self.n_vert = j['num_vertical_tiles']
        self.non_ovlap_px = j['non_overlap_size']
        self.ovlp_px = j['overelap_size']
        self.tiles = [BBox(t) for t in j['tiles']]
    
    def get_tile(self, idx):
        return self.tiles[idx]
    
                
class UnitTransform:
    def __init__(self, micron_to_mosaic_matrix_path):
        self.um_to_px = np.loadtxt(micron_to_mosaic_matrix_path)
        self.px_to_um = np.linalg.inv(self.um_to_px)
        
    # @classmethod
    # def _transform_geo(self, geom, m):
    #     return affine_transform(geom, [*m[:2, :2].flatten(), *m[:2, 2].flatten()])
    
    @classmethod
    def _transform(self, geo_or_xy, m):
        if isinstance(geo_or_xy, geometry.base.BaseGeometry):
            return affine_transform(geo_or_xy, [*m[:2, :2].flatten(), *m[:2, 2].flatten()])
        else:
            _xy1 = np.array(geo_or_xy)
            _xy1 = np.concatenate((_xy1, np.ones((_xy1.shape[0], 1))), axis=1)
            return np.matmul(_xy1, m.T)[:, :2]

    def mosaic_to_micron(self, geo_or_xy):
        return self._transform(geo_or_xy, self.px_to_um)
    
    def micron_to_mosaic(self, geo_or_xy):
        return self._transform(geo_or_xy, self.um_to_px)

class Transcripts:
    def __init__(self, ori_dir, proc_dir, unit_transform,
                 ori_fn='detected_transcripts.csv',
                 proc_fn='transcripts.npz',):
        proc_fn = Path(proc_dir)/proc_fn
        ori_fn = Path(ori_dir)/ori_fn
        self.unit_transform = unit_transform
        if proc_fn.exists():
            logger.info('Loading transcript coord file: transcripts.npz')
            npz = np.load(proc_fn, allow_pickle=True)
            self.coords_micron = npz['coords_micron']
            self.genes = npz['genes']
        else:
            logger.info('Transcript coord file not found. Preparing it. This may take a while.')
            logger.info('Reading original transcripts')
            df = pd.read_csv(ori_fn, index_col=0).sort_values(['global_x', 'global_y', 'global_z'])
            self.coords_micron = df[['global_x','global_y', 'global_z']].values.astype(np.float32)
            self.genes = df['gene'].values
 
            logger.info('Writing transcript file transcripts.npz')
            np.savez(proc_fn, 
                     coords_micron = self.coords_micron, 
                     genes = self.genes)

    def get_transcripts(self, bbox, z=None, genes=None):
        x,y,w,h = bbox
        (xmin,ymin),(xmax,ymax) = self.unit_transform.mosaic_to_micron(np.array([[x,y],[x+w,y+h]]))
        # logger.debug(f'px bbox:{bbox}')
        # logger.debug(f'um bounds: {(xmin,xmax,ymin,ymax)}')
        within = (self.coords_micron[:,0]>=xmin) &\
                 (self.coords_micron[:,0]<=xmax) &\
                 (self.coords_micron[:,1]>=ymin) &\
                 (self.coords_micron[:,1]<=ymax)
        if z is not None:
            if isinstance(z,int):
                within &= self.coords_micron[:,2]==z
            else:
                within &= np.isin(self.coords_micron[:,2], z)
        if isinstance(genes, str):
            within &= self.genes==genes
        elif isinstance(genes, (list, tuple)):
            within &= np.isin(self.genes, genes)
        elif genes is not None:
            raise ValueError(f'Invalid input for genes: {genes}')
        
        coords = self.coords_micron[within]
        genes = self.genes[within]
        df = pd.DataFrame(np.hstack((coords, genes[:,np.newaxis])), 
                          columns=['global_x','global_y','global_z','gene'],
                          index=np.where(within)[0])
        return df

    def get_transcripts_mosaic(self, bbox, z=None, genes=None):
        df = self.get_transcripts(bbox, z, genes)
        # df[['global_x','global_y', 'global_z']] = np.matmul(df[['global_x','global_y', 'global_z']].values, self.unit_transform.um_to_px.T)
        df[['global_x','global_y']] = self.unit_transform.micron_to_mosaic(df[['global_x','global_y']].values)

        # coords_micron, genes = self.get_transcripts_micron(bbox, z)
        # coords_mosaic = np.matmul(coords_micron, self.unit_transform.um_to_px.T)
        # return coords_mosaic, genes
        return df

    

        
class MerfishRegion:
    cell_by_gene_file = 'cell_by_gene.csv'
    cell_meta_file = 'cell_metadata.csv'
    cell_boundary_file = 'cell_boundaries.parquet'
    
    def __init__(self, region_dir):

        self._prepare_dir(region_dir)
        
        self.region_id = self.region_dir.name
        
        self._read_manifest()
        
        self._prepare_tiles()

        logger.info('Preparing unit transformer')
        self.unit_transform = UnitTransform(self.img_dir/'micron_to_mosaic_pixel_transform.csv')

        logger.info('Preparing transcripts')
        self.transcripts = Transcripts(self.region_dir, self.morefish_dir, self.unit_transform)

        #self.detected_transcripts = pd.read_csv(self.region_dir/'detected_transcripts.csv', index_col=0)
        #self.detected_transcripts = gpd.GeoDataFrame(self.detected_transcripts,
        #                                             geometry=self.detected_transcripts.apply(\
        #                                                     lambda x:geometry.Point(x['global_x'],x['global_y']), 
        #                                            axis=1))
        #self.detected_transcripts = self.detected_transcripts.rename(columns={'geometry':'position_micron'})
        #self.detected_transcripts['position_mosaic'] = self.detected_transcripts['position_micron']\
        #                                            .apply(self.unit_transform.micron_to_mosaic)

    def __repr__(self) -> str:
        return (f"MerfishRegion("
                f"region_dir='{self.region_dir}', "
                f"region_id='{self.region_id}')")

    def _prepare_dir(self, region_dir):
        logger.info('Preparing directories')
        self.region_dir = Path(region_dir)
        self.img_dir = self.region_dir/'images'
        
        self.morefish_dir = self.region_dir/'MOREFISH'
        self.morefish_dir.mkdir(exist_ok=True)
        
        self.reseg_dir = self.morefish_dir/'ReSeg'
        self.reseg_dir.mkdir(exist_ok=True)

    def _read_manifest(self):
        logger.info('Reading MERSCOPE manifest')
        with open(self.img_dir/'manifest.json') as f:
            self.manifest = json.load(f)
            
        self.region_width = self.manifest['mosaic_width_pixels']
        self.region_height = self.manifest['mosaic_height_pixels']
#         self.n_hor_tiles = self.manifest['hor_num_tiles_box']
#         self.n_vert_tiles = self.manifest['vert_num_tiles_box']
        self.mosaic_files = pd.DataFrame(self.manifest['mosaic_files'])

    def _prepare_tiles(self):
        self.tiles = Tiles(self.region_width, self.region_height)
#         self.tiles.write_tiles_file(str(self.morefish_dir/'tiles.json'))
        try:
            self.tiles.read_tiles_file(str(self.morefish_dir/'tiles.json'))
            logger.info('Tile file tiles.json loaded')
        except FileNotFoundError:
            self.tiles.write_tiles_file(str(self.morefish_dir/'tiles.json'))
            logger.info('Tile file tiles.json not found. Creating a new one')

    def list_seg_results(self):
        names = [x for x in self.reseg_dir.glob('**/') if x!=self.reseg_dir]
        names = ['native'] + [x.name for x in names]
        seg_names = []
        for name in names:
            if self.check_seg_results(name):
                seg_names.append(name)
        return seg_names

    def check_seg_results(self, name=None):
        cxg_fn, cmeta_fn, cb_fn = self._seg_name_to_path(name)
        return cxg_fn.exists() and cmeta_fn.exists() and cb_fn.exists()

    def _seg_name_to_path(self, name):
        if name is not None and name!='native':
            cxg_fn = self.reseg_dir/name/self.cell_by_gene_file
            cmeta_fn = self.reseg_dir/name/self.cell_meta_file
            cb_fn = self.reseg_dir/name/self.cell_boundary_file
        else:
            cxg_fn = self.region_dir/self.cell_by_gene_file
            cmeta_fn = self.region_dir/self.cell_meta_file
            cb_fn = self.region_dir/self.cell_boundary_file
        return cxg_fn, cmeta_fn, cb_fn       
        
    def mosaic_file_name(self, stain, z):
        try:
            name = self.mosaic_files.query(f'stain=="{stain}" and z=={z}')['file_name']\
                        .values[0]
        except KeyError:
            raise ValueError(f'Stain {stain} or z-plane {z} not found in manifest')
        return name
    
    def mosaic_file_path(self, stain, z):
        name = self.mosaic_file_name(stain, z)
        return self.img_dir/name
    
    def get_stain_image(self, stain, z, bbox=None):
        if isinstance(stain, str):
            stains = [stain]
        elif len(stain)>1 and bbox is None:
            logger.warning('Multiple stains are provided, but no bbox is provided. '
                        'If you want to read the whole region, '
                        'please provide the bbox of the whole region explicitly, '
                        f'which is [0, 0, {self.region_width}, {self.region_height}].'
                        'Reading the whole region may result in large memory usage and long running time.')
            raise ValueError('Multiple stains are provided, but no bbox is provided')
        else:
            stains = stain

        img = np.zeros((*bbox[2:][::-1], len(stains)), dtype=np.uint16)
        for i, stain in enumerate(stains):
            if stain is not None:
                tif_fn = self.mosaic_file_path(stain, z)
                img[:,:,i] = tifffile.memmap(tif_fn)[bbox.y:bbox.y+bbox.h, bbox.x:bbox.x+bbox.w]
        img = np.squeeze(img)
        return img

    def get_stain_image_rgb(self, stain, z, box=None):
        if isinstance(stain, str):
            stains = [stain, None, None]
        elif len(stain)<=3:
            stains = list(stain) + [None]*(3-len(stain))
        else:
            raise ValueError(f'More than 3 stains are provided: {stain}')
        return self.get_stain_image(stains, z, box)

    # def get_transcripts(self, bbox, z=None, genes=None):

        

    # def read_single_stain_image(self, stain, z, bbox=None):
    #     tif_fn = self.mosaic_file_path(stain, z)
    #     img = tifffile.memmap(tif_fn)
    #     if bbox is not None:
    #         wi,hi,ws,hs = bbox
    #         return img[hi:hi+hs, wi:wi+ws]
    #     else:
    #         return img
    
    # def read_multi_stains_image(self, bbox, stains=['DAPI','PolyT',None], z=3):
    #     img = np.zeros((*bbox[2:][::-1], 3))
    #     if isinstance(stains, str):
    #         stains = [stains, None, None]
    #     for i, stain in enumerate(stains):
    #         if stain is not None:
    #             img[:,:,i] = self.read_single_stain_image(stain, z, bbox, )
    #     return img

    def get_tile_image(self, stains, z, tile_idx):
        bbox = self.tiles.tiles[tile_idx]
        return self.get_stain_image(stains, z, bbox)
    
    def get_tile_image_rgb(self, stains, z, tile_idx):
        bbox = self.tiles.tiles[tile_idx]
        return self.get_stain_image_rgb(stains, z, bbox)
    

    def _save_boundaries(self, seg_gdf, name, override=False):
        rlt_dir = self.reseg_dir/name
        rlt_dir.mkdir(exist_ok=True)
        cb_fn = rlt_dir/self.cell_boundary_file
        geo_col = 'geometry' if 'geometry' in seg_gdf.columns else 'Geometry'
        if cb_fn.exists() and not override:
            logger.warning(f'{cb_fn} already exists. Use "override=True" if you want to update it')
            return
        _cb = gpd.GeoDataFrame(columns=['ID', 'EntityID', 'ZIndex', 'Geometry', 'Type', 
                                        'ZLevel', 'Name', 'ParentID', 'ParentType'],
                                geometry='Geometry')
        _cb['ID'] = np.arange(len(seg_gdf))
        _cb['EntityID'] = seg_gdf.index.values
        _cb['Geometry'] = seg_gdf[geo_col].values
        _cb['Type'] = 'cell'
        ##TODO
        # _cb['ZIndex'] = None
        # _cb['ZLevel'] = None
        # _cb['Name'] = None
        # _cb['ParentID'] = None
        # _cb['ParentType'] = None
        _cb.to_parquet(cb_fn)

    def _save_meta(self, seg_gdf, cxg_df, name, override=False):
        rlt_dir = self.reseg_dir/name
        rlt_dir.mkdir(exist_ok=True)
        cmeta_fn = rlt_dir/self.cell_meta_file
        geo_col = 'geometry' if 'geometry' in seg_gdf.columns else 'Geometry'
        if cmeta_fn.exists() and not override:
            logger.warning(f'{cmeta_fn} already exists. Use "override=True" if you want to update it')
            return
        _cmeta = pd.DataFrame(index = seg_gdf.index, columns = ['fov', 'volume', 'center_x', 'center_y', 
                                                                'min_x', 'min_y', 'max_x', 'max_y', 
                                                                'anisotropy', 'transcript_count',
                                                                'perimeter_area_ratio', 'solidity', 
                                                                'DAPI_raw', 'DAPI_high_pass', 
                                                                'PolyT_raw', 'PolyT_high_pass'])
        xx,yy=np.array(seg_gdf[geo_col].centroid.apply(lambda x: (x.x, x.y)).to_list()).T
        _cmeta['center_x'] = xx
        _cmeta['center_y'] = yy
        _cmeta.loc[:,['min_x','min_y','max_x','max_y']] = seg_gdf[geo_col].bounds.values
        _cmeta['transcript_count'] = cxg_df.sum(1).reindex(_cmeta.index).fillna(0)
        _cmeta.to_csv(cmeta_fn)

    def _save_cellbygene(self, cxg_df, name, override=False):
        rlt_dir = self.reseg_dir/name
        rlt_dir.mkdir(exist_ok=True)
        
        cxg_fn = rlt_dir/self.cell_by_gene_file
        if cxg_fn.exists() and not override:
            logger.warning(f'{cxg_fn} already exists. Use "override=True" if you want to update it')
            return
        cxg_df.to_csv(cxg_fn)


    def save_reseg_results(self, seg_gdf, cell_x_gene_df, name, override=False):
        self._save_boundaries(seg_gdf, name, override)
        self._save_meta(seg_gdf, cell_x_gene_df, name, override)
        self._save_cellbygene(cell_x_gene_df, name, override)


    ### TODO
    
#     def save_reseg_results(self, seg_gdf, cell_x_gene_df, name, override=False):
#         rlt_dir = self.reseg_dir/name
#         rlt_dir.mkdir(exist_ok=True)
        
#         cmeta_fn = rlt_dir/self.cell_meta_file
#         cxg_fn = rlt_dir/self.cell_by_gene_file
#         cb_fn = rlt_dir/self.cell_boundary_file
#         if cmeta_fn.exists() and not override:
#             raise ValueError(f'{cmeta_fn} already exists. Use "override=True" if you want to update it')
#         if cxg_fn.exists() and not override:
#             raise ValueError(f'{cxg_fn} already exists. Use "override=True" if you want to update it')
#         if cb_fn.exists() and not override:
#             raise ValueError(f'{cb_fn} already exists. Use "override=True" if you want to update it')
        
#         cols = pd.read_csv(self.region_dir/self.cell_by_gene_file, index_col=0, nrows=2).columns
#         _cxg = cell_x_gene_df.reindex(columns=cols).fillna(0)
        
        
#         _cb = gpd.GeoDataFrame(columns=['ID', 'EntityID', 'ZIndex', 'Geometry', 'Type', 
#                                         'ZLevel', 'Name', 'ParentID', 'ParentType'],
#                                geometry='Geometry')
#         _cb['ID'] = np.arange(len(seg_gdf))
#         _cb['EntityID'] = seg_gdf.index.values
#         _cb['Geometry'] = seg_gdf['geometry'].values
#         _cb['Type'] = 'cell'
# #         _cb = _cb.fillna(None)

        
#         _cmeta = pd.DataFrame(index = _cb['EntityID'], columns = ['fov', 'volume', 'center_x', 'center_y', 
#                                                                   'min_x', 'min_y', 'max_x', 'max_y', 
#                                                                   'anisotropy', 'transcript_count',
#                                                                   'perimeter_area_ratio', 'solidity', 
#                                                                   'DAPI_raw', 'DAPI_high_pass', 
#                                                                   'PolyT_raw', 'PolyT_high_pass'])
#         xx,yy=np.array(_cb['Geometry'].centroid.apply(lambda x: (x.x, x.y)).to_list()).T
#         _cmeta['center_x'] = xx
#         _cmeta['center_y'] = yy
#         _cmeta.loc[:,['min_x','min_y','max_x','max_y']] = _cb['Geometry'].bounds.values
#         _cmeta['transcript_count'] = cell_x_gene_df.sum(1).reindex(_cmeta.index).fillna(0)
        
#         _cmeta.to_csv(rlt_dir/self.cell_meta_file)
#         _cxg.to_csv(rlt_dir/self.cell_by_gene_file)
#         _cb.to_parquet(rlt_dir/self.cell_boundary_file)
        

    def load_reseg_results(self, name=None):
        def _negcols(cxg):
            return cxg.columns[cxg.columns.str.startswith('Blank-')]

        if name is None:
            name = 'native'
        if not self.check_seg_results(name):
            raise ValueError(f'{name} does not refer to a valid segmentation result')
        cxg_fn, cmeta_fn, cb_fn = self._seg_name_to_path(name)

        cxg = pd.read_csv(cxg_fn, index_col=0, dtype={'cell': str,'EntityID': str})
        cmeta = pd.read_csv(cmeta_fn, index_col=0, dtype={'cell': str,'EntityID': str})
        cb = gpd.read_parquet(cb_fn)
        cb.index = cb['EntityID']
        cb = cb.reindex(cxg.index)
        
        negcxg = cxg.loc[:,cxg.columns.isin(_negcols(cxg))]
        cxg = cxg.loc[:,~cxg.columns.isin(_negcols(cxg))]
        
        adata = anndata.AnnData(cxg, obs=cmeta.reindex(cxg.index), dtype=int)
        adata.uns['neg_probs'] = negcxg
        adata.uns['cell_boundary'] = cb[['Geometry']]
    #     adata.uns['merfish_region'] = self
        return adata#, cxg, cmeta, cb
    
