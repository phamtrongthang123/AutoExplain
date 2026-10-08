#!/usr/bin/env python3
"""Run the predefined real Gemma4 study. No LLM API calls, no random-model fallback."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-download',action='store_true',help='Download the pinned ~10.25GB checkpoint if needed')
    parser.add_argument('--device',default='cuda:0',choices=['cuda:0','cpu'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists; choose a new report path')
    os.environ.setdefault('USE_TF','0')
    import torch
    import transformers
    from autoexplain.pretrained import load_gemma4
    from autoexplain.pretrained_study import run_pretrained_study
    torch.set_num_threads(2)
    torch.manual_seed(7)
    start=time.perf_counter()
    report={'status':'failed'}
    try:
        if args.device.startswith('cuda') and torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        bundle=load_gemma4(allow_download=args.allow_download,decoder_device=args.device)
        report=run_pretrained_study(bundle)
        print(json.dumps({'status':report['status'],'pairs':len(report['pairs']),
                          'experiment_seconds':report['experiment_seconds']},indent=2))
    except Exception as exc:
        report.update(error_type=type(exc).__name__)
        raise
    finally:
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report.update(total_seconds=time.perf_counter()-start,
                      cpu_peak_rss_gib=rss/(1024**3 if sys.platform=='darwin' else 1024**2),
                      python_version=sys.version.split()[0],torch_version=str(torch.__version__),
                      transformers_version=transformers.__version__,threads=2)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as file:json.dump(report,file,indent=2,allow_nan=False)


if __name__=='__main__':main()
