import shutil
from pathlib import Path
from tqdm import tqdm

TRAIN_DIR = Path(r'D:\mfa_products\New_products\Aligned_textgrids\timbuck_train_aligned')

for p in tqdm(list(TRAIN_DIR.rglob('*.TextGrid'))):
    base = p.name
    new_path = TRAIN_DIR / base
    if not new_path.is_file():
        shutil.copy(p, new_path)