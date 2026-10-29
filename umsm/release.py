"""Reproducible source handoff; never packages data, model weights or credentials."""
import hashlib
import json
from pathlib import Path
import re
import zipfile


def package_source(root, output):
    # 输入：工程根目录、ZIP 路径；输出：含源码哈希清单的交接包，不含凭据和实验资产；继承：无（模块函数）。
    root,output = Path(root).resolve(),Path(output).resolve()
    if output.suffix.lower()!='.zip':
        raise ValueError('Source package output must end in .zip')
    files = list(root.glob('*.py'))
    files += [root/name for name in ('README.md','requirements.txt','.gitignore','.env.example')]
    for name in ('umsm','configs','tests','md','.github'):
        files += [p for p in (root/name).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml','.md','.txt')]
    selected = sorted({p for p in files if p.exists() and '__pycache__' not in p.parts})
    manifest = []
    for path in selected:
        if not path.resolve().is_relative_to(root):
            raise ValueError('A source symlink leaves the project directory')
        raw = path.read_bytes()
        if re.search(rb'sk-[A-Za-z0-9]{24,}',raw):
            raise ValueError(f'Possible embedded credential: {path.relative_to(root)}; remove it before sharing')
        manifest.append({'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()})
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path,'UMSM/'+path.relative_to(root).as_posix())
        archive.writestr('UMSM/SOURCE_MANIFEST.json',json.dumps({'format':'umsm-source-v1',
            'entrypoint':'python run.py','files':manifest,
            'excluded':['.env','data','models','cache','calibration','results','.umsm','dist']},indent=2))
    print(f'Source package: {output} ({len(selected)} files)')
    return str(output)
