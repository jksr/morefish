from pathlib import Path
from ..merfish import MerfishRegion
from ..seg.seg import Segmentor, CellPoseModel

def is_region_dir(fn):
    return fn.strip('/').split('/')[-1].startswith('region_')

class morefish_cli:
    def segment(self, exp_or_region_dir, seg_name, ##mr args
                model_type=None, pretrained_path=None, gpu=True, ##seg args
                stains='DAPI+PolyT+', z=3, override=False, diameter=None, channels='1,2', simplify_eps=0.5,
                merge_ratio = 0.5,

               ):
        if is_region_dir(exp_or_region_dir):
            reg_dirs = [exp_or_region_dir]
        else:
            reg_dirs = [str(p) for p in Path(exp_or_region_dir).glob('region_*')]


        if ((model_type is None) and (pretrained_path is None)) or ((model_type is not None) and (pretrained_path is not None)):
            raise ValueError('Either model_type or pretrained_path should be provided.')
        
        model = model = CellPoseModel(model_type=model_type, pretrained_path=pretrained_path, gpu=gpu)

        stains = [ None if x == '' else x for x in stains.split('+')]
        channels = [ int(x) for x in channels.split(',')]

        for reg_dir in reg_dirs:
            mr = MerfishRegion(reg_dir)
            segor = Segmentor(mr, seg_name, model=model)
            segs = segor.segment_all_tiles(stains=stains, z=z, override=override, diameter=diameter, channels=channels, simplify_eps=simplify_eps)
            cell_segs = segor.compile_segs(segs, merge_ratio=merge_ratio)
            cxg = segor.partition_transcripts(cell_segs, override=override)
            mr.save_reseg_results(cell_segs, cxg, seg_name)

            

