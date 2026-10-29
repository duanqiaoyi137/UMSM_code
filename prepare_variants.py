"""Additional MCAR/MAR/MNAR or natural-mask transplantation with an unmasked reference."""
import argparse
import hashlib
import itertools
from pathlib import Path

from umsm.data.loader import load_dataset
from umsm.data.masking import base_sample, mask_frame, stable_seed
from umsm.data.preparation import publish_task
from umsm.calibration import set_fingerprint

ROOT = Path(__file__).resolve().parent


def prepare(args, root=ROOT):
    # 输入：CLI 参数与项目路径；输出：派生受控任务的 provenance 列表；继承：无（模块函数）。
    if not args.name.replace('_','').replace('-','').isalnum() or min(args.rows)<1 or args.min_rows<1 or not 0<=args.dense_max_missing<=1:
        raise ValueError('Use safe output name and positive row counts')
    donors = []
    if 'transplant' in args.modes:
        if not args.donor_dataset:
            raise ValueError('Specify --donor-dataset for transplant')
        donors = [t for s in load_dataset(args.donor_dataset,str(root/'data'),sets=args.donor_sets)
                  for t in s.tables]
    manifests = []
    for source in load_dataset(args.dataset,str(root/'data'),sets=args.sets):
        for row_cap,seed in itertools.product(args.rows,args.seeds):
            min_columns = 2 if 'mar' in args.modes else 1
            oracle = base_sample(source,row_cap,seed,args.dense_max_missing,args.complete_rows,args.min_rows,min_columns)
            for mode in args.modes:
                for rate in ([args.rates[0]] if mode=='transplant' else args.rates):
                    frames, audits = {}, {}
                    for index,table in enumerate(oracle.tables):
                        donor = donors[index%len(donors)] if mode=='transplant' else None
                        frame, _, audit = mask_frame(table.df,mode,rate,stable_seed(seed,table.name),
                                                       donor.df if donor is not None else None)
                        if donor is not None:
                            audit['donor_table'] = donor.name
                        frames[table.name],audits[table.name] = frame,audit
                    case = f'{source.slug}_{mode}_r{rate:g}_n{row_cap}_s{seed}'
                    protocol = {'adapter_version':1,'mechanism':mode,'rate':None if mode=='transplant' else rate,
                        'row_cap':row_cap,'mask_seed':seed,'source_dataset':args.dataset,'source_set':source.name,
                        'source_fingerprint':set_fingerprint(source),'oracle_fingerprint':set_fingerprint(oracle),
                        'dense_max_missing':args.dense_max_missing,'complete_rows':args.complete_rows,
                        'min_rows':args.min_rows,'min_columns':min_columns,'donor_dataset':args.donor_dataset,'donor_sets':args.donor_sets,
                        'donor_mask_hashes':{t.name:hashlib.sha256(t.df.isna().to_numpy().tobytes()).hexdigest() for t in donors},
                        'meaning':'additional masking of observed real cells; oracle is pre-mask input, not test-tuned decisions'}
                    manifest = publish_task(root/'data'/args.name/case,frames,oracle.groundtruth,protocol,
                        variants={'oracle':{t.name:t.df for t in oracle.tables}},cell_masks=audits)
                    manifests.append(manifest)
                    print(args.name+'/'+case, 'tables=',len(frames),'GT pairs=',manifest['gt_pairs'])
    return manifests


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--sets',nargs='+')
    parser.add_argument('--name',required=True)
    parser.add_argument('--modes',nargs='+',choices=('mcar','mar','mnar','transplant'),default=['mcar','mar','mnar'])
    parser.add_argument('--rates',nargs='+',type=float,default=[.2,.4,.6,.8])
    parser.add_argument('--rows',nargs='+',type=int,default=[10,30,100,300])
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--dense-max-missing',type=float,default=1.)
    parser.add_argument('--complete-rows',action='store_true')
    parser.add_argument('--min-rows',type=int,default=2)
    parser.add_argument('--donor-dataset')
    parser.add_argument('--donor-sets',nargs='+')
    prepare(parser.parse_args())
