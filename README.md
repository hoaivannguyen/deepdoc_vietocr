
# *Deep*Doc - Công cụ OCR nhanh và tiết kiệm chi phí

- [1. Introduction](#1)
- [2. Vision](#2)
- [3. Parser](#3)

<a name="1"></a>

## Giới thiệu

Với một loạt tài liệu từ nhiều nguồn khác nhau với nhiều định dạng khác nhau và cùng với các yêu cầu truy xuất đa dạng,  một công cụ trích xuất chính xác là rất cần thiết với bất kỳ doanh nghiệp nào. Hôm nay mình xin phép giới thiệu công cụ Deepdoc, một công cụ OCR rất nhanh và tiết kiệm chi phí khi chỉ cần chạy trên CPU. Không những vậy còn có các tính năng kèm theo là Layout Detection (nhận diện bố cục) và Table Structure Recognition (nhận diện và trích xuất bảng) giúp giữ định dạng văn bản sau OCR. 

Tuy nhiên Deepdoc chưa được chuẩn hóa cho tiếng Việt nên mình đã thay VietOCR và bản ONNX vào phần Text Recognizer để có thể nhận dạng văn bảng tiếng Việt tốt hơn. Bạn cũng có thể tham khảo deepdoc phiên bản gốc tại [đây](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md).

Một số cài đặt trước khi chạy chương trình:
```bash
python t_ocr.py -h
usage: t_ocr.py [-h] --inputs INPUTS [--output_dir OUTPUT_DIR]

options:
  -h, --help            hiển thị thông báo trợ giúp này và thoát
  --inputs INPUTS       Thư mục lưu trữ hình ảnh hoặc tệp PDF hoặc đường dẫn tệp đến một hình ảnh hoặc tệp PDF duy nhất
  --output_dir OUTPUT_DIR
                        Thư mục lưu trữ hình ảnh đầu ra. Mặc định: './ocr_outputs'
```
```bash
python t_recognizer.py -h
usage: t_recognizer.py [-h] --inputs INPUTS [--output_dir OUTPUT_DIR] [--threshold THRESHOLD] [--mode {layout,tsr}]

options:
  -h, --help            hiển thị thông báo trợ giúp này và thoát
  --inputs INPUTS       Thư mục lưu trữ hình ảnh hoặc tệp PDF hoặc đường dẫn tệp đến một hình ảnh hoặc tệp PDF duy nhất
  --output_dir OUTPUT_DIR
                        Thư mục lưu trữ hình ảnh đầu ra. Mặc định: './layouts_outputs'
  --threshold THRESHOLD
                        Ngưỡng để lọc ra các phát hiện. Mặc định: 0.5
  --mode {layout,tsr}   Chế độ tác vụ: nhận dạng bố cục (layout) hoặc nhận dạng cấu trúc bảng (tsr)
```

<a name="2"></a>

## 1. OCR
  OCR là một giải pháp rất thiết yếu và cơ bản, thậm chí là phổ biến để trích xuất văn bản. Hãy chạy lệnh sau để thử nghiệm OCR
    ```bash
        python deepdoc/vision/t_ocr.py --inputs=path_to_images_or_pdfs --output_dir=path_to_store_result
     ```
    Đầu vào có thể là thư mục chứa hình ảnh hoặc PDF, hoặc một hình ảnh hoặc PDF. Đầu ra sẽ gồm 1 ảnh với các bounding box được nhận diện và 1 file txt chứa văn bản được OCR.
    <div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\Screenshot 2025-08-28 171633.png" width="900"/>
    </div>

    Mình đang để mặc định là VietOCR Seq2seq vì hiện đang chạy tương đối nhanh và chính xác. Bạn có thể đổi sang VietOCR Transformer trong module/ocr.py nhưng mình không đề xuất vì thời gian xử lý lâu hơn rất nhiều mà độ chuẩn xác không tănng lên là mấy. Nếu bạn muốn nhanh nhất có thể chuyển sang sử dụng bản ONNX bằng việc import ocr_onnx thay vì ocr nhưng độ chính xác sẽ giảm đi 1 chút.

## 2. Layout Recognizer (Nhận diện bố cục)
  - Nhận dạng bố cục. Tài liệu từ các lĩnh vực khác nhau có thể có nhiều bố cục khác nhau,
    ví dụ, báo, tạp chí, sách và sơ yếu lý lịch có bố cục khác nhau.
    Phần nhận dạng này được gán 10 loại nhãn bố cục cơ bản bao phủ hầu hết các trường hợp:
      - Văn bản
      - Tiêu đề
      - Hình ảnh
      - Chú thích hình ảnh
      - Bảng
      - Chú thích bảng
      - Đầu đề
      - Chân trang
      - Tài liệu tham khảo
      - Phương trình

     Hãy thử lệnh sau để xem kết quả Layout Recognizer.
     ```bash
        python deepdoc/vision/t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=layout --output_dir=path_to_store_result
     ```
    Đầu vào có thể là thư mục chứa hình ảnh hoặc PDF, hoặc một hình ảnh hoặc PDF. Đầu ra sẽ gồm 1 ảnh với các gán nhãn như dưới đây:
    <div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\49806-Article Text-153529-1-10-20200804_page-0002.jpg" width="1000"/>
    </div>

## 3. Table Structure Recognizer
  - Nhận dạng cấu trúc bảng (Table Structure Recognition - TSR). Bảng dữ liệu là một cấu trúc thường được sử dụng để trình bày dữ liệu bao gồm số hoặc văn bản.
    Và cấu trúc của một bảng có thể rất phức tạp, như tiêu đề phân cấp, ô trải dài và tiêu đề hàng được chiếu..
    Có năm nhãn cho nhiệm vụ TSR:
      - Cột
      - Hàng
      - Đầu đề cột
      - Đầu đề hàng được chiếu
      - Ô trải dài

    Hãy thử lệnh sau để xem kết quả TSR.
     ```bash
        python deepdoc/vision/t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=tsr --output_dir=path_to_store_result
     ```
    Đầu vào có thể là thư mục chứa hình ảnh hoặc PDF, hoặc một hình ảnh hoặc PDF. Đầu ra sẽ là 1 ảnh với gán nhãn và 1 file markdown với nội dung bảng
    <div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\Screenshot 2025-08-28 182132.png" width="1000"/>
    </div>

## Kết
Hy vọng các bạn thấy công cụ hữu ích và áp dụng được vào thực tế. Nếu có góp ý hãy để lại dưới phần bình luận. Cảm ơn các bạn đã đọc bài viết! 
<a name="3"></a>


    