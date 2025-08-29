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

import logging
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
from module import LayoutRecognizer, TableStructureRecognizer, init_in_out
import argparse
import re
import numpy as np

from datetime import datetime

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "t_recognizer.log")

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
    images, outputs = init_in_out(args)
    if args.mode.lower() == "layout":
        detr = LayoutRecognizer("layout")
        layouts = detr.forward(images, thr=float(args.threshold))
        for i, lyt in enumerate(layouts):
            print(f"Image {i}:")
            for region in lyt:
                print(f"  Detected label: {region.get('type', '')}")
    if args.mode.lower() == "tsr":
        detr = TableStructureRecognizer()
        ocr = OCR()
        layouts = detr(images, thr=float(args.threshold))
    for i, lyt in enumerate(layouts):
        if args.mode.lower() == "tsr":
            # lyt = [t for t in lyt if t["type"] == "table column"]
            markdown = get_table_markdown(images[i], lyt, ocr)
            with open(outputs[i] + ".md", "w+", encoding='utf-8') as f:
                f.write(markdown)
            lyt = [{
                "type": t["label"],
                "bbox": [t["x0"], t["top"], t["x1"], t["bottom"]],
                "score": t["score"]
            } for t in lyt]
        img = draw_box(images[i], lyt, detr.labels, float(args.threshold))
        img.save(outputs[i], quality=95)
        logging.info("save result to: " + outputs[i])


def get_table_markdown(img, tb_cpns, ocr):
    boxes = ocr(np.array(img))
    boxes = LayoutRecognizer.sort_Y_firstly(
        [{"x0": b[0][0], "x1": b[1][0],
          "top": b[0][1], "text": t[0],
          "bottom": b[-1][1],
          "layout_type": "table",
          "page_number": 0} for b, t in boxes if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
        np.mean([b[-1][1] - b[0][1] for b,_ in boxes]) / 3
    )

    def gather(kwd, fzy=10, ption=0.6):
        nonlocal boxes
        eles = LayoutRecognizer.sort_Y_firstly(
            [r for r in tb_cpns if re.match(kwd, r["label"])], fzy)
        eles = LayoutRecognizer.layouts_cleanup(boxes, eles, 5, ption)
        return LayoutRecognizer.sort_Y_firstly(eles, 0)

    headers = gather(r".*header$")
    rows = gather(r".* (row|header)")
    spans = gather(r".*spanning")
    clmns = sorted([r for r in tb_cpns if re.match(
        r"table column$", r["label"])], key=lambda x: x["x0"])
    clmns = LayoutRecognizer.layouts_cleanup(boxes, clmns, 5, 0.5)

    for b in boxes:
        ii = LayoutRecognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
        if ii is not None:
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]
            b["H"] = ii

        ii = LayoutRecognizer.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii

    markdown = TableStructureRecognizer.construct_table(boxes, markdown=True)
    return markdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Thư mục lưu trữ hình ảnh hoặc tệp PDF hoặc đường dẫn tệp đến một hình ảnh hoặc tệp PDF duy nhất",
                        required=True)
    parser.add_argument('--output_dir', help="Thư mục lưu trữ hình ảnh đầu ra. Mặc định: './layouts_outputs'",
                        default="./layouts_outputs")
    parser.add_argument(
        '--threshold',
        help="Ngưỡng để lọc ra các phát hiện. Mặc định: 0,5",
        default=0.5)
    parser.add_argument('--mode', help="Chế độ tác vụ: nhận dạng bố cục (layout) hoặc nhận dạng cấu trúc bảng (tsr)", choices=["layout", "tsr"],
                        default="layout")
    args = parser.parse_args()
    main(args)




    
