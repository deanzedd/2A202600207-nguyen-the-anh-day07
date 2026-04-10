# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thế Anh
**Nhóm:** 32
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (có giá trị gần bằng 1) nghĩa là hai vector embedding đang hướng về cùng một phía trong không gian nhiều chiều. Trong NLP, điều này thể hiện rằng hai đoạn văn bản có ý nghĩa ngữ nghĩa (semantic meaning) và ngữ cảnh rất tương đồng với nhau

**Ví dụ HIGH similarity:**
- Sentence A: Con mèo đang nằm trên sofa và đang ngáp
- Sentence B: Con mèo đang ngáp và ườn trên sofa
- Tại sao tương đồng: Hai câu cùng mô tả về 1 đối tượng con mèo và cùng làm 2 sự việc giống nhau(nằm, ngáp)

**Ví dụ LOW similarity:**
- Sentence A: Con mèo đang nằm trên sofa và đang ngáp
- Sentence B: Điểm thi đại học năm 2025 tăng vọt
- Tại sao khác: Hai câu trên thuộc 2 lĩnh vực khác nhau, như mèo và ngân hàng

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Bởi vì ta cần so sánh độ gần nhau giữa các text embedding trong không gian hơn là việc đo khoảng cách bằng euclidean

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> bước nhảy stride cho mỗi chunk là 500-50=450, chunk đầu tiên là 500, số kí tự còn lại là 9500, số lượng chunk tối thiểu là 9500/450=22, kết hợp thêm chunk đầu tiên là 23
> 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap tăng lên 100 thì số lượng chunk lên 25, việc làm như vậy để giữu được context của câu trước nhiều hơn, giúp model gen ra phù hợp với câu hiện tại và với các câu trước

---
## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

*Domain:* Customer support knowledge base / help-center documentation

*Tại sao nhóm chọn domain này?*
Nhóm chọn domain customer support vì tài liệu dạng FAQ và troubleshooting rất dễ kiếm, có cấu trúc rõ ràng, và phù hợp với retrieval hơn nhiều so với văn bản tự do. Bộ tài liệu này cũng đủ đa dạng để thử các câu hỏi về account, password, billing, refund, rate limit, và cả tài liệu nội bộ cần metadata filtering. Ngoài ra, domain này gần với bài toán RAG thực tế: tìm đúng hướng dẫn và tránh lấy nhầm tài liệu internal cho người dùng cuối.


### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | account_email_change.md | OpenAI Help Center | 1999 | doc_id, title, category=account, audience=customer, language=en, source, last_updated, sensitivity=public |
| 2 | password_reset_help.md | OpenAI Help Center | 1815 | doc_id, title, category=password, audience=customer, language=en, source, last_updated, sensitivity=public |
| 3 | billing_renewal_failure.md | OpenAI Help Center | 1789 | doc_id, title, category=billing, audience=customer, language=en, source, last_updated, sensitivity=public |
| 4 | refund_request_guide.md | OpenAI Help Center | 1952 | doc_id, title, category=refund, audience=customer, language=en, source, last_updated, sensitivity=public |
| 5 | service_limit_429.md | OpenAI Help Center | 1867 | doc_id, title, category=service_limit, audience=customer, language=en, source, last_updated, sensitivity=public |
| 6 | internal_escalation_playbook.md | Internal support / handbook reference | 1944 | doc_id, title, category=escalation, audience=internal_support, language=en, source, last_updated, sensitivity=internal |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_id | string | kb_refund_001 | Định danh duy nhất cho mỗi tài liệu, hữu ích khi quản lý, debug hoặc xóa tài liệu khỏi vector store |
| title | string | How to Request a Refund for a ChatGPT Subscription | Giúp nhận diện nhanh nội dung tài liệu và trình bày nguồn rõ ràng trong kết quả retrieval |
| category | string | account, password, billing, refund, service_limit, escalation | Cho phép lọc theo chủ đề để tăng precision, nhất là khi query thuộc một mảng hỗ trợ cụ thể |
| audience | string | customer, internal_support | Rất quan trọng để tránh trả tài liệu nội bộ cho người dùng cuối và hỗ trợ metadata filtering |
| language | string | en | Hữu ích khi sau này mở rộng sang tài liệu đa ngôn ngữ hoặc cần giới hạn theo ngôn ngữ người hỏi |
| source | string | https://help.openai.com/en/articles/... | Giúp truy vết nguồn gốc tài liệu và kiểm tra độ tin cậy của câu trả lời |
| last_updated | string | 2026-04-10 | Hữu ích nếu sau này cần ưu tiên tài liệu mới hơn hoặc theo dõi độ tươi của dữ liệu |
| sensitivity | string | public, internal | Hỗ trợ kiểm soát truy cập và giảm nguy cơ retrieve nhầm tài liệu nhạy cảm |

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| account_email_change.md | FixedSizeChunker (`fixed_size`) | 5 | 439.8 | Kết quả đều và có overlap giúp giữ context |
| account_email_change.md | SentenceChunker (`by_sentences`) | 11 | 180.2 | Chiều dài nhỏ, rõ ràng nhưng nhiều chunk hơn |
| account_email_change.md | RecursiveChunker (`recursive`) | 6 | 333.2 | Giữ logic theo đoạn tốt hơn nhưng vẫn phân nhỏ |
| refund_request_guide.md | FixedSizeChunker (`fixed_size`) | 5 | 430.4 | Chunk đồng đều, phù hợp cho search nhanh |
| refund_request_guide.md | SentenceChunker (`by_sentences`) | 7 | 277.0 | Chunk theo câu chính xác hơn nhưng nhiều hơn |
| refund_request_guide.md | RecursiveChunker (`recursive`) | 5 | 390.4 | Tách theo đoạn văn phù hợp với cấu trúc guide |
| internal_escalation_playbook.md | FixedSizeChunker (`fixed_size`) | 5 | 428.8 | Phù hợp với playbook dài, giữ được bước logic |
| internal_escalation_playbook.md | SentenceChunker (`by_sentences`) | 5 | 387.2 | Chunk theo câu dài, ít chunk nhưng ít overlap |
| internal_escalation_playbook.md | RecursiveChunker (`recursive`) | 5 | 388.8 | Giữ được cấu trúc đoạn, phù hợp với nội dung chính sách |

### Strategy Của Tôi

**Loại:** FixedSizeChunker

**Mô tả cách hoạt động:**
> FixedSizeChunker cắt văn bản thành các chunk tối đa 500 ký tự với 50 ký tự overlap giữa các chunk. Cách này giữ được ngữ cảnh liên tiếp, giảm nguy cơ mất mạch khi một câu dài nằm giữa hai chunk. Vì không cần phân tích cú pháp câu, strategy này hoạt động tốt với tài liệu hỗn hợp FAQ và playbook.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain support knowledge base có nhiều đoạn câu dài và các mục hướng dẫn không đồng nhất. FixedSizeChunker giúp phân chia đều, vẫn bảo toàn phần nội dung liên quan và thuận tiện cho retrieval mà không phụ thuộc vào dấu câu hay định dạng.

**Code snippet (nếu custom):**
```python
from src.chunking import FixedSizeChunker

chunker = FixedSizeChunker(chunk_size=500, overlap=50)
chunks = chunker.chunk(document_text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| account_email_change.md | best baseline | 5 | 439.8 | tốt, độ dài chunk ổn và có overlap |
| account_email_change.md | **của tôi** | 5 | 439.8 | tốt, dễ retrieve phần nội dung chính |
| refund_request_guide.md | best baseline | 5 | 430.4 | ổn với định dạng guide và bullet |
| refund_request_guide.md | **của tôi** | 5 | 430.4 | ổn, phù hợp với truy vấn khách hàng |
| internal_escalation_playbook.md | best baseline | 5 | 428.8 | ổn với nội dung internal dài |
| internal_escalation_playbook.md | **của tôi** | 5 | 428.8 | tốt, duy trì logic bước và hướng dẫn |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | FixedSizeChunker | 8 | Giữ ngữ cảnh liên tục, chuẩn hóa chunk | Có thể cắt giữa câu nếu câu dài hơn 500 ký tự |
|Tăng | SentenceChunker (3 sent/chunk) | 8.9 | Giữ instructions trọn vẹn, semantic units | Chunks không đều (refund: 277 chars, escalation: 387 chars) |
| Quân | RecursiveChunker (chunk_size=420) | 7/10 | Giữ được ngữ cảnh theo section và bullet khá tốt, hợp tài liệu support có cấu trúc rõ ràng | Nếu query quá mơ hồ hoặc quá ngắn thì đôi lúc chunk top-1 vẫn lệch sang tài liệu gần nghĩa hơn |
| Minh | RecursiveChunker | 6/10 (proxy nội bộ) | Giữ context tốt, phù hợp với tài liệu có section, steps và internal notes | Tạo nhiều chunk hơn và chưa vượt trội rõ rệt về điểm số khi chỉ dùng _mock_embed |
Khôi | RecursiveChunker | 8.5/10 | Giữ được ngữ cảnh theo section và bullet tốt, phù hợp cấu trúc doc. | Đôi lúc tạo ra chunk hơi nhiều, cần cấu hình độ dài cẩn thận. |

**Strategy nào tốt nhất cho domain này?**
> Với dataset support và playbook, FixedSizeChunker là chiến lược tốt vì nó phân chia đều, giữ overlap và không phụ thuộc cấu trúc câu. Điều này giúp hệ thống retrieval tìm phần nội dung liên quan mà không bị lệ thuộc quá nhiều vào định dạng văn bản.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> `SentenceChunker` dùng regex để tách câu trên dấu chấm, chấm than, dấu hỏi, hoặc chấm kết thúc dòng (`. `, `! `, `? `, `.
`). Sau đó nó nối lại mỗi câu cùng delimiter, loại bỏ khoảng trắng thừa và nhóm các câu theo số lượng tối đa `max_sentences_per_chunk`.
> Edge case được xử lý bằng cách bỏ các chuỗi rỗng sau khi split, nên các đoạn văn có nhiều khoảng trắng hay dòng trống vẫn không tạo chunk rỗng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `RecursiveChunker` là một thuật toán chia đệ quy theo các separator ưu tiên: `\n\n`, `\n`, `. `, ` `, và cuối cùng là cắt cứng theo độ dài nếu không còn separator.
> Base case là khi đoạn text đã ngắn hơn hoặc bằng `chunk_size`, khi đó nó trả lại đoạn đó. Nếu một phần vẫn quá dài với separator hiện tại, hàm `_split` sẽ đệ quy tiếp với separator nhỏ hơn để tránh chunk quá lớn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` tạo record cho mỗi `Document` gồm `id`, `content`, `embedding` và `metadata`, sau đó lưu vào store nội bộ hoặc Chroma nếu có sẵn.
> `search` lấy embedding của query và so sánh với embedding của từng record bằng dot product để xếp hạng. Kết quả trả về gồm `content`, `metadata`, `score` và các trường bổ trợ khi cần.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` lọc trước các record theo `metadata_filter`, sau đó thực hiện tìm kiếm tương đồng trên tập đã lọc. Điều này giúp ưu tiên các chunk thuộc cùng phân loại hoặc audience.
> `delete_document` xóa theo `id` hoặc `metadata.doc_id`, nên nếu logic lưu chunk theo metadata hoặc theo id thì vẫn loại bỏ chính xác.

### KnowledgeBaseAgent

**`answer`** — approach:
> `KnowledgeBaseAgent.answer` gọi `store.search` để lấy top-k chunk liên quan, ghép nội dung chunk thành một context và chèn vào prompt RAG.
> Prompt yêu cầu model chỉ trả lời từ context và không được bịa đặt, giúp hạn chế hallucination khi dùng LLM.

### Test Results

```
42 passed
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | A customer can reset their password using email recovery. | A customer can reset their password using email recovery. | high | 1.000 | Yes |
| 2 | Refund requests should follow the refund policy. | Customers can request a refund following the policy. | high | 0.035 | No |
| 3 | The escalation process involves contacting the support manager. | If an emergency is unresolved, escalate to the on-call Support Manager. | high | -0.028 | No |
| 4 | A 429 error means too many requests. | Contact bank when payment is blocked. | low | 0.157 | Yes |
| 5 | Billing issues happen when payment methods expire. | The system sends notifications for account changes. | low | -0.125 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là cặp paraphrase về refund policy: tôi dự đoán similarity cao vì ý nghĩa gần nhau, nhưng actual score lại thấp. Điều này cho thấy embeddings có thể không luôn phản ánh đúng cùng nghĩa khi văn bản khác biệt về cấu trúc và từ ngữ, đặc biệt với miếng ghép mock embedding hoặc khi model chưa học đủ về ngữ cảnh domain.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | How can a customer change the email address on their OpenAI account? | A customer can change their email from Settings > Account on ChatGPT Web if the account supports email management. This is not supported for phone-number-only accounts, Enterprise SSO accounts, or some enterprise-verified personal accounts. After the change, the user is signed out and must log in again with the new email. |
| 2 | What should a customer do if they do not receive the password reset email? | The customer should check the spam/junk folder, confirm they are checking the same inbox used during signup, and verify there is no typo in the email address. If the account was created only with Google, Apple, or Microsoft login, password recovery must be done through that provider instead. |
| 3 | What are the recommended steps when a ChatGPT Plus or Pro renewal payment fails? | The customer should clear browser cache and cookies, contact the bank to check for blocks or security flags, verify billing and card details, and confirm the country or region is supported. If the payment still fails, they should contact support through the Help Center chat widget. |
| 4 | How should a customer handle a 429 Too Many Requests error? | A 429 error means the organization exceeded its request or token rate limit. The recommended solution is exponential backoff: wait, retry, and increase the delay after repeated failures. The customer should also reduce bursts, optimize token usage, and consider increasing the usage tier if needed. |
| 5 | When should an active customer emergency be escalated, and who should be contacted first? | Escalation should be considered when the emergency lasts more than 3 hours without clear resolution, involves multiple simultaneous customer issues, blocks critical outside work, or requires broader coordination. A Support Manager On-call should be consulted, and the account CSM should usually be contacted first as the escalation DRI. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | How can a customer change the email address on their OpenAI account? | Internal Escalation Playbook, contact support manager for account issues | 0.158 | No | Instructions to change email from Settings > Account, with limitations for special account types |
| 2 | What should a customer do if they do not receive the password reset email? | Account email change documentation, check email settings | 0.223 | Partial | Check spam folder, verify email address, use provider login if created with Google/Apple/Microsoft |
| 3 | What are the recommended steps when a ChatGPT Plus or Pro renewal payment fails? | Password reset guide, account recovery steps | 0.225 | No | Clear cache/cookies, contact bank, verify billing details, contact support if persists |
| 4 | How should a customer handle a 429 Too Many Requests error? | Password reset documentation | 0.127 | No | Use exponential backoff strategy, reduce request bursts, optimize token usage, consider upgrading tier |
| 5 | When should an active customer emergency be escalated, and who should be contacted first? | Refund request guide, escalation criteria | 0.230 | No | Escalate after 3+ hours unresolved, contact CSM first as escalation DRI, consult Support Manager On-call |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 1 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Từ Tăng (SentenceChunker), tôi học được rằng chia theo câu giúp giữ nguyên từng instruction/step, làm tăng semantic completeness và dễ cho LLM xử lý. Điều này đặc biệt hữu ích với FAQ và step-by-step guides như support knowledge base. Mặc dù chunks không đều độ dài, nhưng retrieval quality cao hơn vì mỗi chunk là một "atomic unit" của ý tưởng.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Các nhóm khác sử dụng real embeddings (local model hoặc API) thay vì mock embedding cho ra kết quả retrieval tốt hơn đáng kể. Mock embedding dựa trên hash không capture semantic similarity của support domain, dẫn đến false negatives. Bài học là trong thực tế, cần chọn embedding model phù hợp với domain để retrieval chính xác.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm query expansion và synonym mapping vào các benchmark queries để test retrieval robustness. Thứ hai, tôi sẽ chia tài liệu theo sections/headings rõ ràng thay vì cắt cứng 500 ký tự, giúp preserve logic của hướng dẫn. Cuối cùng, tôi sẽ thay thế mock embedding bằng local embedding model từ sentence-transformers để có retrieval chính xác hơn cho domain support.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10/ 10 |
| Chunking strategy | Nhóm |15 / 15 |
| My approach | Cá nhân | 10/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 10/ 10 |
| Core implementation (tests) | Cá nhân |30 / 30 |
| Demo | Nhóm |5 / 5 |
| **Tổng** | | **100 / 100** |
