# Phần mở rộng "Your Turn" — 5/5

Tất cả 5 extension trong `Guide.md §10` đều đã được thực hiện, chạy được và đo lường bằng số.
Kiểm tra tự động vẫn xanh sau khi sửa: `python verify.py` → 11/11, `pytest -q` → 28 passed
(15 test gốc của lab **không bị sửa**, cộng 13 test mới cho logic extension trong
`tests/test_extensions.py`).

Chạy toàn bộ: `python missions/run_all.py`

| # | Extension | File đã sửa / tạo | Kết quả đo được |
|---|---|---|---|
| 1 | Cải thiện `recommend_tier()` | `finops/pricing.py`, `missions/m3_purchasing.py` | 39.1% → **39.4%** tiết kiệm (−$72/tháng) |
| 2 | Right-sizing theo MBU | `missions/m1_efficiency_audit.py` | **$792/tháng** từ 2 GPU |
| 3 | `cache_is_worth_it()` | `finops/pricing.py`, `missions/m2_inference_levers.py` | hoà vốn **0.28 đọc/ghi**, thực đo **0.65** |
| 4 | Ngân sách reasoning | `missions/m2_inference_levers.py`, `missions/m5_report.py` | 8.4% traffic = **94% điện năng** |
| 5 | Carbon-aware scheduling | `missions/m6_carbon_scheduling.py` (mới) | **−1,701 kgCO2e/tháng (92%)** |

---

## Extension 1 — `recommend_tier()` biết so giá

**Vấn đề của chính sách gốc.** Nó dùng đúng hai giả định, cả hai đều sai:

1. Một tỷ lệ thu hồi spot phẳng 5%/giờ cho mọi loại GPU.
2. Bất cứ thứ gì "steady" đều đặt reserved 3 năm.

**Đã thêm.**

- `SPOT_INTERRUPT_RATE` — tỷ lệ thu hồi theo từng loại GPU. H100 3%/giờ, A10G 12%/giờ,
  L4 15%/giờ. Phần cứng khan hiếm được thuê theo cụm dài hạn nên ít bị cắt; card
  inference phổ thông thì dung lượng mỏng và churn liên tục.
- `recommend_tier()` nhận thêm `gpu_type`, `spot_hr`, `on_demand_hr`, `reserved_hr`. Khi
  có đủ giá, nó **tính thật**: mô phỏng rework theo tỷ lệ thu hồi của chính loại GPU đó,
  quy ra đơn giá spot hiệu dụng, rồi so với tier ổn định. Spot không còn được chọn chỉ vì
  job có thể checkpoint.
- `recommend_commit_term()` — tách quyết định *kỳ hạn cam kết* khỏi quyết định *tier*.
  Duty cycle trả lời "khi chạy thì có bận không"; persistence (`days / days_in_month`)
  trả lời "nó có chạy suốt không". Chỉ job đứng gần như cả tháng mới đáng khoá 3 năm.

Hàm giữ nguyên hợp đồng cũ khi không truyền giá, nên `test_recommend_tier` gốc vẫn pass.

**Kết quả.**

| Chính sách | Chi phí tối ưu/tháng | % tiết kiệm |
|---|---|---|
| Gốc (5%/giờ phẳng, luôn 3yr) | $15,627 | 39.1% |
| Mới (rủi ro theo GPU + kỳ hạn theo độ bền) | $15,555 | 39.4% |

**Tại sao policy mới cho kết quả khác?** Bốn job đổi quyết định. Ba job training
(`job-train-llm`, `job-train-embed`, `job-finetune`) chạy 3–14 ngày nên không còn được
gán reserved 3 năm — chúng chuyển sang spot không cam kết, và vì là H100 (3%/giờ thay vì
5%/giờ) nên chi phí rework giảm. Ngược lại `job-dev-sandbox` chạy trên A10G bị đánh giá
rủi ro cao hơn (12%/giờ) nên *đắt lên* $6 — đúng hướng, vì rủi ro đó có thật.

Chênh lệch $72/tháng nhỏ, nhưng giá trị nằm ở chỗ nó **đúng vì lý do đúng**. Ma trận
quyết định mới cho thấy điều chính sách cũ không bao giờ làm được:

```
  gpu        risk         3h/int       8h/int      20h/int    20h/fixed    24h/fixed
  H100        3%           spot         spot     reserved     reserved     reserved
  A100        5%           spot         spot     reserved     reserved     reserved
  A10G       12%           spot         spot         spot     reserved     reserved
  L4         15%           spot         spot         spot     reserved     reserved
```

Với H100, spot $1.50/giờ **thua** reserved-3yr $1.40/giờ ở duty cycle cao — chính sách
mới từ chối spot. Chính sách cũ luôn chọn spot vì nó không hề so giá. Với A10G và L4,
chiết khấu spot sâu hơn (60%) nên spot vẫn thắng dù rủi ro gấp bốn lần.

---

## Extension 2 — Right-sizing theo MBU, không theo `$/GPU-hr`

**Đã thêm** vào `missions/m1_efficiency_audit.py`:

- `dollars_per_gb_vram()` — đơn giá phía bộ nhớ cho toàn bộ catalog.
- `rightsize_by_mbu()` — đề xuất GPU thay thế cho card trượt **cả hai** ngưỡng
  (MBU < 0.60 **và** MFU < 0.40), tức không bão hoà ở bất kỳ trục nào.
- M1 giờ theo dõi thêm `achieved_tflops`, `achieved_bw_tbs` và `peak_mem_gb` (VRAM đỉnh
  thực đo) cho mỗi GPU.

**Kết quả.**

| GPU | Đổi sang | MBU | VRAM đỉnh | $/hr | Tiết kiệm |
|---|---|---|---|---|---|
| `gpu-h100-4` | H100 → MI300X | 0.207 | 67.0 GB | $2.50 → $1.95 | $396/tháng |
| `gpu-h100-5` | H100 → MI300X | 0.271 | 66.3 GB | $2.50 → $1.95 | $396/tháng |
| | | | | | **$792/tháng** |

Bảng `$/GB-VRAM` xếp hạng khác hẳn `$/GPU-hr`:

| GPU | $/hr | HBM | **$/GB-hr** |
|---|---|---|---|
| MI300X | $1.95 | 192 GB | **$0.0102** |
| A100 | $1.79 | 80 GB | $0.0224 |
| H100 | $2.50 | 80 GB | $0.0312 |
| L4 | $0.80 | 24 GB | $0.0333 |
| A10G | $1.00 | 24 GB | $0.0417 |

**Tại sao không chỉ chọn GPU rẻ nhất theo `$/GPU-hr`?** Vì giá mỗi giờ không nói lên GPU
đó có *chạy nổi* workload hay không. L4 rẻ nhất bảng ($0.80/giờ) nhưng đắt gần nhất tính
theo GB VRAM, và băng thông chỉ 0.30 TB/s. Thiếu VRAM thì job OOM — tiết kiệm bằng 0.
Thiếu băng thông thì với workload memory-bound, thời gian chạy dài ra đúng tỷ lệ: job
chạy lâu gấp đôi trên card rẻ bằng nửa giá thì tổng chi phí **không đổi**, chỉ có latency
tệ đi.

Vì vậy bộ lọc là ba điều kiện cứng — đủ `peak_bw_tbs` để đạt MBU mục tiêu, đủ
`peak_tflops_fp16` để đạt MFU mục tiêu, đủ `hbm_gb` cho VRAM đỉnh cộng 20% dự phòng — rồi
mới chọn card rẻ nhất trong tập khả thi. Bộ lọc này thật, không phải hình thức:
`gpu-a10g-1` bị loại vì L4 cần 0.302 TB/s mà chỉ có 0.30; hai con A100 bị loại vì không
GPU rẻ hơn nào đủ băng thông.

---

## Extension 3 — `cache_is_worth_it()`

**Đã thêm** vào `finops/pricing.py`: `cache_write_premium()`, `cache_break_even_reads()`,
`cache_is_worth_it()`. Trong `missions/m2_inference_levers.py`: `cache_reuse()` đo mức tái
sử dụng thật từ dữ liệu, và M2 chỉ áp chiết khấu cache khi `cache_is_worth_it()` trả True.

**Cần đọc lại bao nhiêu lần để cache có lợi?**

Ghi cache tốn phụ trội 25% giá input (`CACHE_WRITE_MULTIPLIER = 1.25`); mỗi lần đọc lại
tiết kiệm 90%. Hoà vốn = 0.25 / 0.90 = **0.28 lần đọc mỗi lần ghi**.

| Tier | Giá input | Phụ trội ghi | Hoà vốn |
|---|---|---|---|
| small | $0.20/1M | $0.050/1M | 0.28 đọc |
| large | $3.00/1M | $0.750/1M | 0.28 đọc |

Điểm hoà vốn **giống hệt nhau ở hai tier** — cả phí ghi lẫn tiền tiết kiệm mỗi lần đọc
đều tỉ lệ thuận với giá input nên tỉ số triệt tiêu. Khác nhau là *số tiền đặt cược*:
chênh 15 lần. Cache trên model lớn không dễ có lãi hơn, chỉ là khi có lãi thì lãi to hơn.

**Dataset của chúng ta có đạt ngưỡng không?** Có, nhưng biên mỏng hơn nhiều người tưởng.
Coi mỗi prefix là một bộ `(team, project, route_tier)`; trong mỗi cửa sổ TTL, request đầu
tiên trả phí ghi, phần còn lại là đọc:

| TTL | Prefix | Ghi | Đọc | Đọc/ghi | Kết luận |
|---|---|---|---|---|---|
| 5 phút (mặc định) | 16 | 1,456 | 944 | **0.65** | có lãi, vượt hoà vốn 2.3× |
| 1 giờ (extended) | 16 | 296 | 2,104 | **7.11** | vượt hoà vốn 25× |

Với TTL mặc định, 1,456 trên 2,400 request là *ghi* chứ không phải *đọc* — traffic quá
thưa so với cửa sổ 5 phút, phần lớn cache hết hạn trước khi ai dùng lại. Đổi sang TTL 1
giờ nâng mức tái sử dụng lên 11 lần mà không cần đổi một dòng code ứng dụng nào.

Bài học: **đòn bẩy của cache là tái sử dụng, không phải mức chiết khấu.** Chiết khấu 90%
là hằng số; số lần đọc mới là biến ta điều khiển được — bằng TTL, bằng cách gom traffic
cùng prefix, bằng cách ổn định system prompt.

---

## Extension 4 — Ngân sách reasoning

**Đã thêm** vào M2: tách chi phí `$` và năng lượng `Wh` theo cờ `is_reasoning`, cộng kịch
bản what-if ép trần. M5 đưa "Reasoning budget cap" thành một lever trong report.

**Reasoning chiếm bao nhiêu %?**

| | Request | % traffic | Chi phí | % chi phí | Năng lượng | % năng lượng |
|---|---|---|---|---|---|---|
| Reasoning | 201 | **8.4%** | $1.40/ngày | 16.5% | 29,788 Wh | **94.0%** |
| Thường | 2,199 | 91.6% | $7.09/ngày | 83.5% | 1,888 Wh | 6.0% |

Con số đáng chú ý: **8.4% traffic nhưng 94% điện năng.** Chi phí và năng lượng hoàn toàn
không đi cùng nhau — nhìn hoá đơn thì reasoning chỉ là 16.5%, nhìn công tơ điện thì nó là
gần như toàn bộ.

**Tại sao tốn năng lượng ~80× nhiều hơn?** Giá token tính tuyến tính theo số token, còn
năng lượng thì không:

- Chuỗi sinh dài hơn nhiều (trung bình 6,175 token/query so với 2,861), mà decode là
  memory-bound: **mỗi token sinh ra phải đọc lại toàn bộ trọng số model từ HBM**. Năng
  lượng chủ yếu đi vào di chuyển dữ liệu, không phải phép nhân.
- KV-cache phình theo độ dài chuỗi, nên mỗi token sau lại đắt hơn token trước về băng
  thông đọc.
- Chuỗi dài chạy ở batch hiệu dụng thấp hơn, chi phí cố định mỗi bước bị chia cho ít
  token hơn.

Deck lấy hệ số ~74–86× (`REASONING_ENERGY_MULTIPLIER = 80`) so với query model nhỏ.

**Quy tắc định tuyến đề xuất.** Chỉ bật reasoning khi (a) task thuộc nhóm cần suy luận
nhiều bước — chứng minh, lập kế hoạch, sinh code phức tạp; (b) model thường đã trả lời với
confidence dưới ngưỡng; hoặc (c) lần trả lời đầu bị người dùng từ chối. Không bật mặc định.

**Ép trần xuống 3% traffic** (demote các request reasoning có output dài nhất về mức
median của request thường): tiết kiệm **$0.89/ngày ($27/tháng)** và **22,830 Wh/ngày** —
tức bỏ đi **72% tổng điện năng inference** trong khi chỉ động tới 8.4% số request.

Đây là lever kỳ lạ nhất trong toàn bộ lab: gần như vô nghĩa trên cột đô la, đứng đầu bảng
trên cột carbon. Nếu công ty có mục tiêu phát thải phải báo cáo thì đây là thứ đáng làm
trước, dù nó xếp cuối trong bảng ROI tính bằng tiền.

---

## Extension 5 — Carbon-aware scheduling

**File mới:** `missions/m6_carbon_scheduling.py`. Tính năng lượng theo GPU-hours × công
suất board × PUE 1.15, rồi so 5 vùng trên ba trục: tiền điện, carbon, latency.

5/8 workload là interruptible — chúng đã checkpoint sẵn và đã chấp nhận bị dời, nên đây là
4,861 kWh/tháng (**72% điện năng của fleet**) có thể chuyển vùng mà không cần dự án di trú.

| Vùng | $/kWh | gCO2/kWh | Tiền điện/tháng | tCO2e/tháng | Latency | Điểm cân bằng |
|---|---|---|---|---|---|---|
| **europe-north1** | $0.090 | 30 | $437 | 0.15 | 110 ms | **2.64** |
| us-east-wa | $0.055 | 90 | $267 | 0.44 | 60 ms | 4.00 |
| us-west-2 | $0.070 | 120 | $340 | 0.58 | 70 ms | 5.27 |
| us-east-1 *(hiện tại)* | $0.120 | 380 | $583 | 1.85 | 5 ms | 14.85 |
| europe-central2 | $0.180 | 660 | $875 | 3.21 | 120 ms | 25.27 |

Carbon tiết kiệm được nếu chuyển toàn bộ phần interruptible sang vùng sạch nhất:

| Job | GPU | kWh/tháng | kgCO2e hiện tại | kgCO2e vùng sạch | Tiết kiệm |
|---|---|---|---|---|---|
| job-train-llm | H100 | 3,864 | 1,468.3 | 115.9 | 1,352.4 |
| job-train-embed | A100 | 552 | 209.8 | 16.6 | 193.2 |
| job-finetune | H100 | 290 | 110.1 | 8.7 | 101.4 |
| job-dev-sandbox | A10G | 83 | 31.5 | 2.5 | 29.0 |
| job-batch-eval | H100 | 72 | 27.5 | 2.2 | 25.4 |
| | | | | | **1,701 kgCO2e (92%)** |

**Vùng nào là "tối ưu" thực sự? Phụ thuộc ưu tiên nào của công ty.**

Bước đầu tiên không có đánh đổi gì cả: rời `us-east-1` là thắng trên **cả hai** trục —
vùng hiện tại vừa bẩn hơn 13 lần vừa đắt hơn vùng sạch nhất. Chuyển sang `europe-north1`
cắt 1,701 kgCO2e/tháng **và giảm** $146/tháng tiền điện. Việc này nên làm ngay, không cần
chờ chính sách carbon nào.

Đánh đổi thật chỉ xuất hiện ở bước sau, và nó là **lưới sạch nhất không phải lưới rẻ
nhất**:

- **Ưu tiên tiền → `us-east-wa`** ($0.055/kWh, $267/tháng). Vẫn sạch hơn `us-east-1` bốn
  lần. Là lựa chọn "được cả hai" khi công ty không có cam kết carbon cứng.
- **Ưu tiên carbon → `europe-north1`** (30 gCO2/kWh). Tốn thêm $170/tháng so với
  `us-east-wa` để tránh 292 kgCO2e, tức **$583/tấn CO2e**. Con số này đáng đưa ra bàn họp:
  nó nằm trong khoảng giá tín chỉ carbon chất lượng cao có kiểm chứng (~$100–$600/tấn) —
  và khác tín chỉ ở chỗ nó *không phát thải ngay từ đầu* chứ không bù trừ sau.
- **Cân bằng hai trục → `europe-north1`** (điểm 2.64, chuẩn hoá cả `$/kWh` lẫn `gCO2/kWh`
  theo giá trị tốt nhất).
- **Nên tránh: `europe-central2`.** Đắt nhất *và* bẩn nhất — thua trên mọi trục.

**Trade-off latency.** Vùng sạch nhất đúng là xa người dùng nhất: +105 ms một chiều so với
`us-east-1`. Nhưng nó **không áp dụng cho tập công việc này** — cả 5 job interruptible đều
là training và batch eval, không có người dùng nào đang chờ. Ba job inference phục vụ
người dùng thì giữ nguyên tại `us-east-1`. Đó chính là lý do chỉ dời phần interruptible
chứ không dời cả fleet: latency chỉ là chi phí khi có người ngồi đợi.
