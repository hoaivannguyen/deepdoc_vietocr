#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
import sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from module.seeit import draw_box
from module.ocr import OCR
# ONNX
# from from module.ocr_onnx import OCR
from module import init_in_out
import argparse
import numpy as np
import trio

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2' #2 gpus, uncontinuous
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' #1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '' #cpu

import time
import torch
from PIL import Image

from datetime import datetime

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "t_ocr.log")

# Count previous runs by counting lines that start with "=== Run"
run_count = 1
if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        run_count += sum(1 for line in f if line.startswith("=== Run"))

# Write run header with count and date
with open(log_file, "a", encoding="utf-8") as f:
    f.write(f"\n=== Run {run_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

sys.stdout = open(log_file, "a", encoding="utf-8")
sys.stderr = sys.stdout


def main(args):
    import torch.cuda

    cuda_devices = torch.cuda.device_count()
    limiter = [trio.CapacityLimiter(1) for _ in range(cuda_devices)] if cuda_devices > 1 else None
    ocr = OCR()
    images, outputs = init_in_out(args)

    def __ocr(i, id, img):
        print("Task {} start".format(i))
        start_time = time.time()
        bxs = ocr(np.array(img), id)
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = [{
            "text": t,
            "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]],
            "type": "ocr",
            "score": 1} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]]
        img = draw_box(images[i], bxs, ["ocr"], 1.)
        img.save(outputs[i], quality=95)
        with open(outputs[i] + ".txt", "w+", encoding='utf-8') as f:
            f.write("\n".join([o["text"] for o in bxs]))

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Task {i} done in {elapsed:.2f} seconds")

    async def __ocr_thread(i, id, img, limiter = None):
        if limiter:
            async with limiter:
                print("Task {} use device {}".format(i, id))
                await trio.to_thread.run_sync(lambda: __ocr(i, id, img))
        else:
            __ocr(i, id, img)

    async def __ocr_launcher():
        if cuda_devices > 1:
            async with trio.open_nursery() as nursery:
                for i, img in enumerate(images):
                    nursery.start_soon(__ocr_thread, i, i % cuda_devices, img, limiter[i % cuda_devices])
                    await trio.sleep(0.1)
        else:
            for i, img in enumerate(images):
                await __ocr_thread(i, 0, img)

    trio.run(__ocr_launcher)

    print("OCR hoàn thành!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Thư mục lưu trữ hình ảnh hoặc tệp PDF hoặc đường dẫn tệp đến một hình ảnh hoặc tệp PDF duy nhất",
                        required=True)
    parser.add_argument('--output_dir', help="Thư mục lưu trữ hình ảnh đầu ra. Mặc định: './ocr_outputs'",
                        default="./ocr_outputs")
    args = parser.parse_args()
    main(args)
