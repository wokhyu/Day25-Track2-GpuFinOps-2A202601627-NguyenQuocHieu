# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,390  
**Projected savings:** $12,743  (**47%**)

## Savings by lever

| Lever | Savings (USD) | % of baseline | % of savings |
|---|---|---|---|
| Purchasing (spot/reserved) | $10,112 | 37.3% | 79.4% |
| Inference (cascade/cache/batch) | $1,212 | 4.5% | 9.5% |
| Right-size by MBU | $792 | 2.9% | 6.2% |
| Kill idle GPUs | $600 | 2.2% | 4.7% |
| Reasoning budget cap | $27 | 0.1% | 0.2% |
| **Total** | **$12,743** | **47.0%** | **100%** |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cleanest region: europe-north1 (92% less carbon on shiftable load, 1,701 kgCO2e/month)
- Cheapest power: us-east-wa  |  best $/carbon balance: europe-north1
- Shiftable (interruptible) load: 4,861 kWh/month; moving it to the clean region also saves $146/month of electricity
- Cost of choosing clean over cheapest: $583/tonne CO2e avoided

## Vì sao "GPU-Util" là một lời nói dối

`nvidia-smi` báo `GPU-Util` bằng cách hỏi đúng một câu: *trong khoảng lấy mẫu này,
có kernel nào đang chạy không?* Có kernel là 100%. Kernel đó làm việc gì, chạy
nhanh hay đứng chờ, driver không quan tâm. Nó là đồng hồ đo **thời gian bận**,
không phải thước đo **công việc hoàn thành**.

Bằng chứng nằm ngay trong telemetry của fleet này:

| GPU | GPU-Util | MFU | MBU | Diễn giải |
|---|---|---|---|---|
| `gpu-h100-4` | 98.2% | 0.194 | 0.207 | SM bận gần như liên tục nhưng tensor core gần như rỗi |
| `gpu-h100-3` | 93.1% | 0.427 | 0.444 | cùng mức util, gấp 2.2x công việc thật |

Hai card cùng loại H100, cùng giá $2.50/giờ, cùng
báo util trên 90% — nhưng một con làm được hơn gấp đôi số FLOP của con kia. Nếu chỉ
nhìn util thì hai con này trông giống hệt nhau trên dashboard.

**Cơ chế gây ra khoảng cách đó** (cả ba đều giữ util ở 100% mà không sinh ra FLOP):

1. **Memory stall.** MBU của `gpu-h100-4` chỉ 0.207 — băng thông HBM
   cũng thấp ngang FLOP. Cả hai trục đều thấp trong khi SM vẫn bận nghĩa là SM đang
   *đứng chờ dữ liệu*, không phải đang bão hoà băng thông. Đây là dấu hiệu của batch
   quá nhỏ hoặc tensor quá bé để lấp đầy pipeline: kernel bị chặn bởi độ trễ, không
   phải bởi thông lượng.
2. **Sai loại kernel.** Các phép elementwise (activation, layernorm, transpose, ép
   kiểu) chạy trên CUDA core chứ không phải tensor core. Chúng giữ SM bận 100% nhưng
   đóng góp gần như 0 TFLOPS FP16. Model không fuse kernel dính rất nặng lỗi này.
3. **Chờ I/O và đồng bộ.** Dataloader chậm, `torch.cuda.synchronize()` giữa các
   step, hoặc all-reduce phải đợi node chậm nhất. GPU quay vòng chờ, util vẫn đếm là bận.

**Hệ quả tiền bạc.** Chi phí trả theo *giờ GPU*, không theo *FLOP*. Ở MFU
0.194, mỗi đô la thuê `gpu-h100-4` chỉ mua được 19%
lượng compute đã trả tiền — 81% còn lại là tiền đốt. Đơn giá
thực tế của công việc chạy trên nó cao gấp 2.2
lần so với chạy trên `gpu-h100-3`, dù hoá đơn hai bên giống hệt nhau.

Đây cũng là lý do idle **không** phải vấn đề lớn nhất. Lãng phí do GPU nằm không chỉ
$600/tháng (2.2% baseline) — nhỏ, dễ thấy, dễ
sửa. Lãng phí thực sự nằm ở các GPU *đang chạy* với MFU thấp: chúng không xuất hiện
trên bất kỳ báo cáo idle nào.

## Right-sizing theo MBU (không phải theo $/GPU-hr)

| GPU | Đổi sang | MBU | VRAM đỉnh | $/hr | Tiết kiệm/tháng |
|---|---|---|---|---|---|
| gpu-h100-4 | H100 -> MI300X | 0.207 | 67 GB | $2.50 -> $1.95 | $396 |
| gpu-h100-5 | H100 -> MI300X | 0.271 | 66 GB | $2.50 -> $1.95 | $396 |

Không chọn GPU rẻ nhất theo `$/GPU-hr` vì giá mỗi giờ không nói lên GPU đó có *chạy
nổi* workload hay không. Một card rẻ hơn nhưng thiếu VRAM sẽ OOM, còn thiếu băng
thông HBM thì thời gian chạy dài ra đúng tỷ lệ — job chạy lâu gấp đôi trên card rẻ
bằng nửa giá thì tổng chi phí không đổi, mà latency thì tệ đi. Vì vậy bộ lọc ở đây là
ba điều kiện cứng: đủ `peak_bw_tbs` để đạt MBU mục tiêu, đủ `peak_tflops_fp16` để đạt
MFU mục tiêu, đủ `hbm_gb` cho mức VRAM đỉnh đã đo cộng 20% dự phòng. Chỉ trong tập
khả thi đó mới chọn card rẻ nhất. `$/GB-VRAM` là thước đo phụ cho biết mình đang trả
bao nhiêu cho *dung lượng nhớ* — chỉ số này xếp hạng khác hẳn `$/GPU-hr`.

## Kinh tế của prompt cache

Cache không miễn phí: ghi cache tốn phụ trội, mỗi lần đọc lại mới tiết kiệm được tiền.
Điểm hoà vốn là **0.28 lần đọc mỗi lần ghi**, và con
số này *giống nhau ở cả hai tier* — phí ghi và phí đọc đều tỉ lệ với giá input, nên tỷ
lệ triệt tiêu. Cái khác nhau giữa small và large là số tiền tuyệt đối đặt cược, chênh
nhau 15 lần.

Đo trên dữ liệu thật, 16 prefix riêng biệt với TTL mặc định 5 phút cho
1,456 lần ghi và 944 lần đọc, tức
**0.65 đọc/ghi** — chỉ vượt điểm hoà vốn
2.3 lần. Cache *có* lãi,
nhưng biên mỏng. Chuyển sang TTL 1 giờ, cùng lượng traffic đó cho
**7.11 đọc/ghi** — gấp
11 lần.
Đòn bẩy của cache là *tái sử dụng*, không phải bản thân mức chiết khấu.

## Ngân sách reasoning: 8% traffic, 94% năng lượng

| | Request | % traffic | Chi phí | % chi phí | Năng lượng | % năng lượng |
|---|---|---|---|---|---|---|
| Reasoning | 201 | 8.4% | $1.40/ngày | 16.5% | 29,788 Wh | 94.0% |
| Thường | 2,199 | 91.6% | $7.09/ngày | 83.5% | 1,888 Wh | 6.0% |

Chi phí và năng lượng **không đi cùng nhau**: reasoning chiếm
16.5% hoá đơn nhưng 94.0% điện năng. Lý do
là giá token tính tuyến tính theo số token, còn năng lượng thì không: một query
reasoning kéo dài chuỗi sinh, mỗi token sinh ra phải đọc lại toàn bộ trọng số model từ
HBM, cộng thêm KV-cache phình theo độ dài. Deck lấy hệ số ~80x
cho mỗi query reasoning so với query nhỏ thông thường.

**Quy tắc định tuyến đề xuất:** chỉ bật reasoning khi (a) task thuộc nhóm cần suy luận
nhiều bước — chứng minh, lập kế hoạch, sinh code phức tạp — hoặc (b) model thường đã
trả lời với confidence dưới ngưỡng, hoặc (c) lần trả lời đầu bị người dùng từ chối.
Không bật mặc định cho toàn bộ traffic. Ép trần xuống 3% traffic
tiết kiệm $0.89/ngày ($27/tháng)
và 22,830 Wh/ngày — tức bỏ đi tới
72% tổng
điện năng phục vụ inference trong khi chỉ động tới 8.4% số request.

## Purchasing: chính sách cũ vs chính sách mới

Chính sách gốc dùng một tỷ lệ thu hồi spot phẳng 5%/giờ cho mọi loại GPU và luôn đặt
reserved 3 năm. Cả hai giả định đều sai theo hai hướng ngược nhau. Tỷ lệ thu hồi thực
tế phụ thuộc độ khan hiếm của phần cứng: H100 hiếm nhưng thuê theo cụm dài hạn nên ít
bị cắt (3%/giờ), còn A10G/L4 là hàng phổ thông, dung lượng mỏng và
churn liên tục (12%/giờ). Còn commitment 3 năm chỉ hợp lý cho dịch vụ
đứng suốt tháng — gán nó cho một job huấn luyện 14 ngày là mua 35 tháng công suất rỗi.

| Chính sách | Chi phí tối ưu/tháng | % tiết kiệm |
|---|---|---|
| Gốc (5%/giờ phẳng, luôn 3yr) | $15,627 | 39.1% |
| Mới (rủi ro theo GPU + kỳ hạn theo độ bền) | $15,555 | 39.4% |

Chênh lệch $72/tháng nhìn nhỏ, nhưng ý nghĩa nằm ở chỗ
khác: nó đúng *vì lý do đúng*. Với H100 giá spot $1.50 so với reserved-3yr $1.40, chính
sách mới biết từ chối spot cho job chạy 20h/ngày dài hạn — điều chính sách cũ không bao
giờ làm được vì nó không hề so giá.

## Thứ tự hành động theo ROI

| # | Đòn bẩy | Tiết kiệm/tháng | % baseline |
|---|---|---|---|
| 1 | Purchasing (spot/reserved) | $10,112 | 37.3% |
| 2 | Inference (cascade/cache/batch) | $1,212 | 4.5% |
| 3 | Right-size by MBU | $792 | 2.9% |
| 4 | Kill idle GPUs | $600 | 2.2% |
| 5 | Reasoning budget cap | $27 | 0.1% |

**Ưu tiên 1 — Purchasing (spot/reserved) ($10,112/tháng).** Đây là khoản lớn nhất và
là thay đổi cấu hình thuần tuý, không cần viết lại model. Làm trước.

**Ưu tiên 2 — Inference (cascade/cache/batch) ($1,212/tháng).** Trong cụm inference, riêng
cascade (định tuyến truy vấn dễ sang model nhỏ) đã chiếm phần lớn: bỏ cascade thì chi
phí inference tăng hơn bốn lần, trong khi bỏ cache hoặc bỏ batch chỉ làm tăng vài phần
trăm. Triển khai cascade trước, cache và batch sau.

**Ưu tiên 3 — Right-size by MBU ($792/tháng).** Cần đo lại workload trước
khi đổi phần cứng, nên chậm hơn, nhưng là khoản tiết kiệm vĩnh viễn.

**Ưu tiên cuối — Reasoning budget cap và Kill idle GPUs.** Nhỏ về tiền nhưng nên làm vì
rẻ: autoscale-to-zero là vài dòng cấu hình, còn trần reasoning tuy chỉ tiết kiệm
$27/tháng lại cắt phần lớn điện năng — giá trị của
nó nằm ở cột carbon chứ không ở cột đô la.

Lưu ý ngược đời cần nói rõ: **`gpu-h100-4` xứng đáng được điều tra trước cả khi
right-size nó.** Nếu MFU thấp là do batch nhỏ hoặc thiếu kernel fusion thì sửa phần
mềm sẽ lấy lại toàn bộ công suất H100 — đáng giá hơn nhiều so với việc hạ cấp phần cứng
xuống card rẻ hơn. Chỉ hạ cấp khi đã xác nhận workload thật sự không cần con H100.

## Bền vững: carbon gắn với tiền điện

| Vùng | $/kWh | gCO2/kWh | Tiền điện/tháng | tCO2e/tháng | Latency |
|---|---|---|---|---|---|
| europe-north1 | $0.090 | 30 | $437 | 0.15 | 110 ms |
| us-east-wa | $0.055 | 90 | $267 | 0.44 | 60 ms |
| us-west-2 | $0.070 | 120 | $340 | 0.58 | 70 ms |
| us-east-1 | $0.120 | 380 | $583 | 1.85 | 5 ms |
| europe-central2 | $0.180 | 660 | $875 | 3.21 | 120 ms |

5 trên 8 workload là interruptible — chúng đã được checkpoint và
đã chấp nhận bị dời, nên đây là 4,861 kWh/tháng
(72% điện năng của fleet) có thể chuyển vùng
mà không cần dự án di trú nào.

Chuyển toàn bộ phần này từ `us-east-1` sang `europe-north1` cắt
**1,701 kgCO2e/tháng (92%)** và đồng
thời **giảm** tiền điện $146/tháng. Không có đánh đổi nào ở
bước này: `us-east-1` vừa bẩn hơn 13
lần vừa đắt hơn. Đây là loại quyết định nên làm ngay, không cần chờ chính sách carbon nào.

Đánh đổi thật chỉ xuất hiện ở bước sau: **lưới sạch nhất không phải lưới rẻ nhất.**
`us-east-wa` có điện $0.055/kWh, rẻ
hơn `europe-north1`, nhưng bẩn hơn ba lần. Chọn `europe-north1` thay vì
`us-east-wa` tốn thêm $170/tháng để tránh
292 kgCO2e — tức
**$583/tấn CO2e**. Con số đó là thứ đáng đưa ra bàn họp: nó
nằm trong khoảng giá tín chỉ carbon chất lượng cao (~$100–$600/tấn cho loại có kiểm
chứng), nên đây là cách giảm phát thải rẻ ngang hoặc rẻ hơn mua tín chỉ — và khác tín
chỉ ở chỗ nó *không phát thải ngay từ đầu* chứ không bù trừ sau.

Vậy vùng nào "tối ưu"? Phụ thuộc công ty ưu tiên gì. Nếu chỉ tối ưu tiền:
`us-east-wa` — vẫn sạch hơn `us-east-1` bốn lần, nên là lựa chọn "được
cả hai" khi không có cam kết carbon cứng. Nếu có mục tiêu phát thải phải báo cáo:
`europe-north1`, trả thêm $170/tháng. Theo điểm cân bằng
chuẩn hoá cả hai trục, `europe-north1` thắng.

Trade-off latency có tồn tại nhưng **không áp dụng cho tập công việc này**: các job
interruptible đều là training và batch eval, không có người dùng nào đang chờ. Ba job
inference phục vụ người dùng thì giữ nguyên tại `us-east-1` — và đó chính là
lý do chỉ dời phần interruptible chứ không dời cả fleet.

_Figures are June-2026 as-of snapshots; re-baseline before acting._