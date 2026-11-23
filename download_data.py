"""Carregamento sob demanda de registros experimentais do CWRU."""
from pathlib import Path
import requests
from scipy.io import whosmat
from core import CLASSES, FILE_IDS

URL = "https://engineering.case.edu/sites/default/files/{id}.mat"

def download(loads=(0,1),dest=Path("data")):
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    successes,errors=[],[]
    for load in loads:
        for label in CLASSES:
            fid=FILE_IDS[label][load]
            target=dest/f"{fid}.mat"
            try:
                if target.exists() and target.stat().st_size>10000:
                    if any(n.endswith("_DE_time") for n,*_ in whosmat(str(target))):
                        successes.append(fid)
                        continue
                with requests.get(URL.format(id=fid),timeout=60,stream=True,
                                  headers={"User-Agent":"VibraSense-Academic-Demo/2.0"}) as response:
                    response.raise_for_status()
                    with target.open("wb") as out:
                        for chunk in response.iter_content(chunk_size=262144):
                            if chunk:
                                out.write(chunk)
                if not any(n.endswith("_DE_time") for n,*_ in whosmat(str(target))):
                    raise ValueError("Arquivo não contém registro DE_time")
                successes.append(fid)
            except Exception as exc:
                target.unlink(missing_ok=True)
                errors.append((fid,str(exc)))
    return successes,errors

if __name__=="__main__":
    ok,failed=download()
    print(f"Sucesso: {ok}; falhas: {failed}")
