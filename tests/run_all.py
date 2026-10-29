"""Run the plain-function test files in isolated processes (no pytest dependency)."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package',help='verify hashes and run offline tests in a clean source ZIP extraction')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
if args.package:
    with tempfile.TemporaryDirectory() as temporary:
        destination=Path(temporary)
        with zipfile.ZipFile(args.package) as archive:
            for name in archive.namelist():
                if not (destination/name).resolve().is_relative_to(destination.resolve()):
                    raise ValueError('Package member leaves extraction directory')
            archive.extractall(destination)
        extracted=destination/'UMSM'
        manifest=json.loads((extracted/'SOURCE_MANIFEST.json').read_text())
        for row in manifest['files']:
            assert hashlib.sha256((extracted/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
        assert not (extracted/'.env').exists() and not (extracted/'cache').exists()
        subprocess.run([sys.executable,str(extracted/'tests/run_all.py')],cwd=extracted,check=True)
        print('Clean source package hashes and offline tests passed.')
else:
    files=sorted((root/'tests').glob('test_*.py'))
    for path in files:
        subprocess.run([sys.executable,str(path)],cwd=root,check=True)
    print(f'All {len(files)} test files passed.')
