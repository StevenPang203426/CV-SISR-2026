import os
import random
from PIL import Image
from torch.utils.data import Dataset
from utils.img import imresize_bicubic, rgb2y
import torchvision.transforms as T


class SRDataset(Dataset):
    def __init__(self, model, hr_dir, scale=2, patch_size=96, augment=True, is_train=True):
        candidates = sorted(
            [
                os.path.join(hr_dir, p)
                for p in os.listdir(hr_dir)
                if p.lower().endswith(('.png', 'jpg', 'jpeg', 'bmp'))
            ]
        )
        # 过滤损坏图片，避免 DataLoader 在 worker 进程中报错中断训练。
        self.hr_paths = []
        bad_files = []
        for path in candidates:
            try:
                with Image.open(path) as img:
                    img.verify()
                self.hr_paths.append(path)
            except Exception:
                bad_files.append(path)

        if bad_files:
            print(f"[SRDataset] Skip {len(bad_files)} unreadable files from {hr_dir}.")
            print(f"[SRDataset] Example bad file: {bad_files[0]}")

        self.model = model
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.is_train = is_train

    def __len__(self):
        if self.is_train:
            return len(self.hr_paths) * 4
        else:
            return len(self.hr_paths)

    def __getitem__(self, idx):
        if self.is_train:
            img_idx = idx // 4
            rotation = (idx % 4) * 90

            hr = Image.open(self.hr_paths[img_idx]).convert('RGB')
            w, h = hr.size
            hr = hr.resize((round(w / 12) * 12, round(h / 12) * 12), Image.BICUBIC)
            hr = hr.rotate(rotation, expand=True)

            w, h = hr.size
            ps = self.patch_size

            if w < ps or h < ps:
                hr = hr.resize((max(ps, w), max(ps, h)), Image.BICUBIC)
                w, h = hr.size

            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            hr = hr.crop((x, y, x + ps, y + ps))

            if self.augment and random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            if self.augment and random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)

        else:
            hr = Image.open(self.hr_paths[idx]).convert('RGB')
            w, h = hr.size
            hr = hr.resize((round(w / 12) * 12, round(h / 12) * 12), Image.BICUBIC)

        lr = imresize_bicubic(hr, self.scale, down=True)
        if self.model == 'srcnn':
            lr = imresize_bicubic(lr, self.scale, down=False)

        # lr = rgb2y(lr)
        # hr = rgb2y(hr)

        to_tensor = T.ToTensor()
        return {'lr': to_tensor(lr), 'hr': to_tensor(hr)}
