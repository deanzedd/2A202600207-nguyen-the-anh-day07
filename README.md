# Ngày 7 — Nền Tảng Dữ Liệu: Embedding & Vector Store

---

## Mục Tiêu

Sau lab này, bạn cần có thể:
- Giải thích cosine similarity và dự đoán điểm tương đồng giữa các văn bản
- Triển khai 3 chiến lược chunking và so sánh ưu nhược điểm
- Xây dựng vector store với search, filter, và delete
- Kết nối knowledge base với agent qua RAG pattern
- Chỉ ra khi nào retrieval giúp ích và khi nào nó thất bại

---

## Cấu Trúc Lab: 2 Pha

### Pha 1 — Cá Nhân: Hoàn Thành solution.py

Mỗi sinh viên **tự mình** hoàn thành tất cả TODO trong `solution/solution.py`. `Document` dataclass và `chunk_fixed_size` đã được implement sẵn làm ví dụ.

### Pha 2 — Nhóm: So Sánh Retrieval Strategy

Nhóm cùng chọn một bộ tài liệu và thống nhất 5 benchmark queries. Mỗi thành viên **thử strategy riêng** (chunking, metadata), chạy cùng queries, rồi **so sánh kết quả trong nhóm** để học từ nhau.

---

## Thiết Lập Môi Trường

```bash
pip install -r requirements.txt
pytest tests/ -v          # Phần lớn tests sẽ FAIL (chưa implement)
```

(Tùy chọn) Cấu hình API key cho embedding thật:
```bash
cp .env.example .env
```

---

## Cấu Trúc Thư Mục

```
├── README.md              ← Bạn đang đọc file này
├── exercises.md           ← Bài tập (4 phần)
├── solution/
│   └── solution.py        ← Code khung (điền TODO)
├── data/                  ← Tài liệu mẫu + tài liệu nhóm (.txt/.md)
├── tests/
│   └── test_solution.py   ← Test suite (30+ tests)
├── report/
│   └── TEMPLATE_REPORT.md ← Báo cáo (1 file/sinh viên)
├── SCORING.md             ← Tiêu chí chấm điểm
└── requirements.txt
```

---

## Các Giai Đoạn Lab

| Giai Đoạn | Hoạt Động |
|-----------|-----------|
| Chuẩn bị tài liệu | Nhóm chọn domain, thu thập tài liệu, chuyển sang .md/.txt |
| Lập trình cá nhân | Warm-up + implement tất cả TODO (cá nhân) |
| Thiết kế strategy | Mỗi người thử strategy riêng, thống nhất 5 queries |
| So sánh trong nhóm | Chạy benchmark, so sánh kết quả, chuẩn bị demo |
| Demo & thảo luận | Trình bày strategy + so sánh, thảo luận liên nhóm |

---

## Nhiệm Vụ Cá Nhân (Pha 1)

### Đã implement sẵn (tham khảo)
- `Document` dataclass — container cho text + metadata
- `chunk_fixed_size` — sliding window chunking

### Cần implement
- `chunk_by_sentences` — chia theo ranh giới câu
- `chunk_recursive` — thử từng separator theo thứ tự
- `compute_similarity` — cosine similarity
- `compare_chunking_strategies` — so sánh 3 chiến lược
- `EmbeddingStore` — wrapper quanh vector store (5 methods)
- `KnowledgeBaseAgent` — RAG pattern agent

---

## Nhiệm Vụ Nhóm (Pha 2) — So Sánh Strategy

1. **Chọn bộ tài liệu** (5-10 docs): FAQ, SOP, policy, internal docs, hoặc domain bất kỳ
2. **Chuyển sang .txt/.md** nếu cần (xem tips trong exercises.md)
3. **Thống nhất 5 benchmark queries** kèm gold answers
4. **Mỗi thành viên thử strategy riêng**: chunking method, tham số, metadata schema
5. **So sánh kết quả trong nhóm**: strategy nào cho retrieval tốt hơn? Tại sao?

---

## Chấm Điểm

Xem chi tiết tại `SCORING.md`. Tóm tắt:

| Phần | Điểm |
|------|------|
| Cá nhân (code + phân tích) | 60 |
| Nhóm (strategy + so sánh) | 40 |
| **Tổng** | **100** |

---

## Sản Phẩm Nộp Bài

1. `solution/solution.py` — hoàn thành tất cả TODO
2. `report/TEMPLATE_REPORT.md` — một báo cáo/sinh viên (gồm cả phần nhóm và cá nhân)

---

## Chạy Kiểm Thử

```bash
pytest tests/ -v
```
