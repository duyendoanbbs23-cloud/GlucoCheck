# 🩺 Diabetes Risk Check

Ứng dụng Streamlit đa ngôn ngữ (Tiếng Việt · English · 中文 · Español) giúp một người trưởng thành, ngay tại nhà,
quyết định **có nên đi xét nghiệm đường huyết hay không**. Dữ liệu: CDC BRFSS 2015 (253.680 phản hồi).

> **Phát biểu quyết định người dùng:** *An adult answering questions at home uses the app to decide whether to get a blood sugar test.*
> Đây là công cụ sàng lọc mang tính giáo dục, **không phải chẩn đoán**.

## Chạy thử

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Các phần của đề bài

| Phần | Trang trong app | Mức |
|---|---|---|
| Dự đoán một ca + hành động gợi ý | 🩺 My check | Core |
| Mô phỏng ngưỡng và chi phí | ⚖️ Threshold & cost | Core |
| Giải thích 3–5 yếu tố chính (hệ số hồi quy) | 🩺 My check | Core |
| Trang About: dữ liệu, metrics, 5 hạn chế, tuyên bố y tế | ℹ️ About | Core |
| Dự đoán hàng loạt từ CSV, tải kết quả | 📄 Batch (CSV) | Bonus |
| Kiểm tra công bằng theo thu nhập / học vấn / giới / tuổi | 🤝 Fairness | Bonus |
| Model card | 📋 Model card | Bonus |
| User testing | [`docs/USER_TESTING.md`](docs/USER_TESTING.md) (mẫu, nhóm tự điền) | Bonus |

## Thiết kế chính

- **Ít câu hỏi nhất có thể:** chọn tiến (forward selection) trên tập validation. Độ chính xác gần như đi ngang sau ~5 câu;
  app dùng 7 câu (tuổi, chiều cao + cân nặng → BMI, sức khỏe chung, huyết áp cao, mỡ máu cao, bệnh tim, khó đi lại).
- **Chống rò rỉ thông tin:** chỉ dùng thông tin người dùng biết ở nhà. Thu nhập, học vấn, giới tính **không** là đầu vào
  mô hình, chỉ dùng để kiểm tra công bằng. Uống rượu nhiều bị loại vì trong dữ liệu nó trông như "yếu tố bảo vệ", dễ gây hiểu lầm.
- **Mô hình:** hồi quy logistic chuẩn hóa → xác suất được hiệu chỉnh tốt (xem biểu đồ calibration trong About), nên
  ngưỡng tối ưu theo chi phí là `chi phí báo động giả ÷ (báo động giả + bỏ sót)`.
- **Trình bày không gây hoang mang:** "khoảng N trên 100 người", ba mức màu dịu (không dùng đỏ), luôn kèm lời nhắc đây không phải chẩn đoán.
- **Thêm ngôn ngữ:** sao chép khối `"en"` trong `i18n.py`, dịch, rồi thêm vào `LANGS`. Khóa thiếu sẽ tự dùng tiếng Anh.

## Cấu trúc

```
app.py        giao diện Streamlit (6 trang)
core.py       tính điểm, giải thích, ngưỡng/chi phí, kiểm tra CSV (không phụ thuộc Streamlit)
viz.py        biểu đồ SVG thuần (không cần thư viện vẽ)
i18n.py       toàn bộ văn bản theo ngôn ngữ
train.py      huấn luyện + xuất model/, data/ (cần scikit-learn)
model/        model.json (hệ số), metrics.json (metrics, công bằng, calibration)
data/         test_scores.csv.gz (dự đoán trên tập test giữ riêng), sample_batch.csv
tests/        kiểm thử core
```

## Huấn luyện lại

Tải `diabetes_binary_health_indicators_BRFSS2015.csv` (Kaggle: *Diabetes Health Indicators Dataset*), rồi:

```bash
pip install -r requirements-train.txt
python train.py path/to/diabetes_binary_health_indicators_BRFSS2015.csv
```

## Triển khai online (Streamlit Community Cloud)

1. Đẩy repo lên GitHub (public).
2. Vào <https://share.streamlit.io> → **New app** → chọn repo, nhánh `main`, file `app.py` → **Deploy**.

## Hạn chế chính

Dữ liệu tự khai từ khảo sát điện thoại Mỹ 2015; không phải chẩn đoán; độ chính xác thấp hơn ở nhóm thu nhập thấp, học vấn thấp
và người lớn tuổi (AUC 0,76 so với 0,82 ở nhóm thu nhập cao). Chi tiết trong trang About và Model card.

---

**English:** a multilingual Streamlit screening app that helps an adult decide whether to get a blood sugar test
(logistic regression on 7 at-home questions, test AUC ≈ 0.81). Includes a cost-based threshold simulator, per-case
explanations, batch CSV scoring, a fairness audit and a model card. Educational use only, not medical advice.
