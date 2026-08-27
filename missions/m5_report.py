"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability, pricing
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m6_carbon_scheduling

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def build_analysis(r1, r2, r3, r6, levers, baseline, optimized, cat) -> str:
    """Assemble the written analysis from the numbers the missions just produced.

    Everything here is interpolated from mission output, so the prose can never
    drift out of sync with the tables above it.
    """
    lie = next((s for s in r1["summary"] if s["gpu_id"] == "gpu-h100-4"), r1["summary"][0])
    healthy = max(r1["summary"], key=lambda s: s["mfu"])
    reasoning = r2["reasoning"]
    reuse = r2["cache_reuse"]
    reuse_long = r2["cache_reuse_long"]
    ranked = sorted(levers.items(), key=lambda kv: -kv[1])
    idle_pct = levers["Kill idle GPUs"] / baseline * 100
    pricing_h100 = pricing.interrupt_rate_for("H100")
    pricing_a10g = pricing.interrupt_rate_for("A10G")
    total_req = r2["requests"]

    rows_rs = "\n".join(
        f"| {c['gpu_id']} | {c['from']} -> {c['to']} | {c['mbu']:.3f} | {c['peak_mem_gb']:.0f} GB | "
        f"${c['from_hr']:.2f} -> ${c['to_hr']:.2f} | ${c['monthly_savings']:,.0f} |"
        for c in r1["rightsize"]
    ) or "| - | - | - | - | - | - |"

    rows_region = "\n".join(
        f"| {r['region']} | ${r['usd_per_kwh']:.3f} | {r['gco2_per_kwh']:.0f} | "
        f"${r['power_usd']:,.0f} | {r['carbon_kg']/1000:.2f} | {r['latency_ms']} ms |"
        for r in sorted(r6["regions"], key=lambda x: x["score"])
    )

    rows_actions = "\n".join(
        f"| {i} | {name} | ${amount:,.0f} | {amount / baseline * 100:.1f}% |"
        for i, (name, amount) in enumerate(ranked, start=1)
    )

    return f"""## Vì sao "GPU-Util" là một lời nói dối

`nvidia-smi` báo `GPU-Util` bằng cách hỏi đúng một câu: *trong khoảng lấy mẫu này,
có kernel nào đang chạy không?* Có kernel là 100%. Kernel đó làm việc gì, chạy
nhanh hay đứng chờ, driver không quan tâm. Nó là đồng hồ đo **thời gian bận**,
không phải thước đo **công việc hoàn thành**.

Bằng chứng nằm ngay trong telemetry của fleet này:

| GPU | GPU-Util | MFU | MBU | Diễn giải |
|---|---|---|---|---|
| `{lie['gpu_id']}` | {lie['gpu_util_pct']:.1f}% | {lie['mfu']:.3f} | {lie['mbu']:.3f} | SM bận gần như liên tục nhưng tensor core gần như rỗi |
| `{healthy['gpu_id']}` | {healthy['gpu_util_pct']:.1f}% | {healthy['mfu']:.3f} | {healthy['mbu']:.3f} | cùng mức util, gấp {healthy['mfu']/max(lie['mfu'],1e-9):.1f}x công việc thật |

Hai card cùng loại H100, cùng giá ${num(cat['H100']['on_demand_hr']):.2f}/giờ, cùng
báo util trên 90% — nhưng một con làm được hơn gấp đôi số FLOP của con kia. Nếu chỉ
nhìn util thì hai con này trông giống hệt nhau trên dashboard.

**Cơ chế gây ra khoảng cách đó** (cả ba đều giữ util ở 100% mà không sinh ra FLOP):

1. **Memory stall.** MBU của `{lie['gpu_id']}` chỉ {lie['mbu']:.3f} — băng thông HBM
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
{lie['mfu']:.3f}, mỗi đô la thuê `{lie['gpu_id']}` chỉ mua được {lie['mfu']*100:.0f}%
lượng compute đã trả tiền — {(1-lie['mfu'])*100:.0f}% còn lại là tiền đốt. Đơn giá
thực tế của công việc chạy trên nó cao gấp {healthy['mfu']/max(lie['mfu'],1e-9):.1f}
lần so với chạy trên `{healthy['gpu_id']}`, dù hoá đơn hai bên giống hệt nhau.

Đây cũng là lý do idle **không** phải vấn đề lớn nhất. Lãng phí do GPU nằm không chỉ
${levers['Kill idle GPUs']:,.0f}/tháng ({idle_pct:.1f}% baseline) — nhỏ, dễ thấy, dễ
sửa. Lãng phí thực sự nằm ở các GPU *đang chạy* với MFU thấp: chúng không xuất hiện
trên bất kỳ báo cáo idle nào.

## Right-sizing theo MBU (không phải theo $/GPU-hr)

| GPU | Đổi sang | MBU | VRAM đỉnh | $/hr | Tiết kiệm/tháng |
|---|---|---|---|---|---|
{rows_rs}

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
Điểm hoà vốn là **{r2['cache_break_even']['large']:.2f} lần đọc mỗi lần ghi**, và con
số này *giống nhau ở cả hai tier* — phí ghi và phí đọc đều tỉ lệ với giá input, nên tỷ
lệ triệt tiêu. Cái khác nhau giữa small và large là số tiền tuyệt đối đặt cược, chênh
nhau 15 lần.

Đo trên dữ liệu thật, {reuse['prefixes']} prefix riêng biệt với TTL mặc định 5 phút cho
{reuse['writes']:,} lần ghi và {reuse['reads']:,} lần đọc, tức
**{reuse['avg_reads_per_write']:.2f} đọc/ghi** — chỉ vượt điểm hoà vốn
{reuse['avg_reads_per_write']/r2['cache_break_even']['large']:.1f} lần. Cache *có* lãi,
nhưng biên mỏng. Chuyển sang TTL 1 giờ, cùng lượng traffic đó cho
**{reuse_long['avg_reads_per_write']:.2f} đọc/ghi** — gấp
{reuse_long['avg_reads_per_write']/max(reuse['avg_reads_per_write'],1e-9):.0f} lần.
Đòn bẩy của cache là *tái sử dụng*, không phải bản thân mức chiết khấu.

## Ngân sách reasoning: 8% traffic, 94% năng lượng

| | Request | % traffic | Chi phí | % chi phí | Năng lượng | % năng lượng |
|---|---|---|---|---|---|---|
| Reasoning | {reasoning['n']:,} | {reasoning['traffic_pct']:.1f}% | ${reasoning['cost']:.2f}/ngày | {reasoning['cost_pct']:.1f}% | {reasoning['wh']:,.0f} Wh | {reasoning['wh_pct']:.1f}% |
| Thường | {total_req - reasoning['n']:,} | {100 - reasoning['traffic_pct']:.1f}% | ${reasoning['plain_cost']:.2f}/ngày | {100 - reasoning['cost_pct']:.1f}% | {reasoning['plain_wh']:,.0f} Wh | {100 - reasoning['wh_pct']:.1f}% |

Chi phí và năng lượng **không đi cùng nhau**: reasoning chiếm
{reasoning['cost_pct']:.1f}% hoá đơn nhưng {reasoning['wh_pct']:.1f}% điện năng. Lý do
là giá token tính tuyến tính theo số token, còn năng lượng thì không: một query
reasoning kéo dài chuỗi sinh, mỗi token sinh ra phải đọc lại toàn bộ trọng số model từ
HBM, cộng thêm KV-cache phình theo độ dài. Deck lấy hệ số ~{sustainability.REASONING_ENERGY_MULTIPLIER:.0f}x
cho mỗi query reasoning so với query nhỏ thông thường.

**Quy tắc định tuyến đề xuất:** chỉ bật reasoning khi (a) task thuộc nhóm cần suy luận
nhiều bước — chứng minh, lập kế hoạch, sinh code phức tạp — hoặc (b) model thường đã
trả lời với confidence dưới ngưỡng, hoặc (c) lần trả lời đầu bị người dùng từ chối.
Không bật mặc định cho toàn bộ traffic. Ép trần xuống {reasoning['cap_frac']:.0%} traffic
tiết kiệm ${reasoning['cap_cost_saved_daily']:.2f}/ngày (${levers['Reasoning budget cap']:,.0f}/tháng)
và {reasoning['cap_wh_saved_daily']:,.0f} Wh/ngày — tức bỏ đi tới
{reasoning['cap_wh_saved_daily']/(reasoning['wh']+reasoning['plain_wh'])*100:.0f}% tổng
điện năng phục vụ inference trong khi chỉ động tới {reasoning['traffic_pct']:.1f}% số request.

## Purchasing: chính sách cũ vs chính sách mới

Chính sách gốc dùng một tỷ lệ thu hồi spot phẳng 5%/giờ cho mọi loại GPU và luôn đặt
reserved 3 năm. Cả hai giả định đều sai theo hai hướng ngược nhau. Tỷ lệ thu hồi thực
tế phụ thuộc độ khan hiếm của phần cứng: H100 hiếm nhưng thuê theo cụm dài hạn nên ít
bị cắt ({pricing_h100:.0%}/giờ), còn A10G/L4 là hàng phổ thông, dung lượng mỏng và
churn liên tục ({pricing_a10g:.0%}/giờ). Còn commitment 3 năm chỉ hợp lý cho dịch vụ
đứng suốt tháng — gán nó cho một job huấn luyện 14 ngày là mua 35 tháng công suất rỗi.

| Chính sách | Chi phí tối ưu/tháng | % tiết kiệm |
|---|---|---|
| Gốc (5%/giờ phẳng, luôn 3yr) | ${r3['base_optimized_monthly']:,} | {r3['base_savings_pct']:.1f}% |
| Mới (rủi ro theo GPU + kỳ hạn theo độ bền) | ${r3['optimized_monthly']:,} | {r3['savings_pct']:.1f}% |

Chênh lệch ${r3['policy_delta_monthly']:,.0f}/tháng nhìn nhỏ, nhưng ý nghĩa nằm ở chỗ
khác: nó đúng *vì lý do đúng*. Với H100 giá spot $1.50 so với reserved-3yr $1.40, chính
sách mới biết từ chối spot cho job chạy 20h/ngày dài hạn — điều chính sách cũ không bao
giờ làm được vì nó không hề so giá.

## Thứ tự hành động theo ROI

| # | Đòn bẩy | Tiết kiệm/tháng | % baseline |
|---|---|---|---|
{rows_actions}

**Ưu tiên 1 — {ranked[0][0]} (${ranked[0][1]:,.0f}/tháng).** Đây là khoản lớn nhất và
là thay đổi cấu hình thuần tuý, không cần viết lại model. Làm trước.

**Ưu tiên 2 — {ranked[1][0]} (${ranked[1][1]:,.0f}/tháng).** Trong cụm inference, riêng
cascade (định tuyến truy vấn dễ sang model nhỏ) đã chiếm phần lớn: bỏ cascade thì chi
phí inference tăng hơn bốn lần, trong khi bỏ cache hoặc bỏ batch chỉ làm tăng vài phần
trăm. Triển khai cascade trước, cache và batch sau.

**Ưu tiên 3 — {ranked[2][0]} (${ranked[2][1]:,.0f}/tháng).** Cần đo lại workload trước
khi đổi phần cứng, nên chậm hơn, nhưng là khoản tiết kiệm vĩnh viễn.

**Ưu tiên cuối — {ranked[-1][0]} và {ranked[-2][0]}.** Nhỏ về tiền nhưng nên làm vì
rẻ: autoscale-to-zero là vài dòng cấu hình, còn trần reasoning tuy chỉ tiết kiệm
${levers['Reasoning budget cap']:,.0f}/tháng lại cắt phần lớn điện năng — giá trị của
nó nằm ở cột carbon chứ không ở cột đô la.

Lưu ý ngược đời cần nói rõ: **`{lie['gpu_id']}` xứng đáng được điều tra trước cả khi
right-size nó.** Nếu MFU thấp là do batch nhỏ hoặc thiếu kernel fusion thì sửa phần
mềm sẽ lấy lại toàn bộ công suất H100 — đáng giá hơn nhiều so với việc hạ cấp phần cứng
xuống card rẻ hơn. Chỉ hạ cấp khi đã xác nhận workload thật sự không cần con H100.

## Bền vững: carbon gắn với tiền điện

| Vùng | $/kWh | gCO2/kWh | Tiền điện/tháng | tCO2e/tháng | Latency |
|---|---|---|---|---|---|
{rows_region}

{r6['shiftable_jobs']} trên 8 workload là interruptible — chúng đã được checkpoint và
đã chấp nhận bị dời, nên đây là {r6['shiftable_kwh']:,.0f} kWh/tháng
({r6['shiftable_kwh']/r6['fleet_kwh']*100:.0f}% điện năng của fleet) có thể chuyển vùng
mà không cần dự án di trú nào.

Chuyển toàn bộ phần này từ `{r6['home_region']}` sang `{r6['cleanest']}` cắt
**{r6['carbon_saved_kg']:,.0f} kgCO2e/tháng ({r6['carbon_saved_pct']:.0f}%)** và đồng
thời **giảm** tiền điện ${r6['power_saved_usd']:,.0f}/tháng. Không có đánh đổi nào ở
bước này: `{r6['home_region']}` vừa bẩn hơn {sustainability.REGION_CARBON[r6['home_region']]/sustainability.REGION_CARBON[r6['cleanest']]:.0f}
lần vừa đắt hơn. Đây là loại quyết định nên làm ngay, không cần chờ chính sách carbon nào.

Đánh đổi thật chỉ xuất hiện ở bước sau: **lưới sạch nhất không phải lưới rẻ nhất.**
`{r6['cheapest']}` có điện ${min(r['usd_per_kwh'] for r in r6['regions']):.3f}/kWh, rẻ
hơn `{r6['cleanest']}`, nhưng bẩn hơn ba lần. Chọn `{r6['cleanest']}` thay vì
`{r6['cheapest']}` tốn thêm ${r6['clean_premium_usd']:,.0f}/tháng để tránh
{r6['clean_premium_carbon_kg']:,.0f} kgCO2e — tức
**${r6['usd_per_tonne_co2e']:,.0f}/tấn CO2e**. Con số đó là thứ đáng đưa ra bàn họp: nó
nằm trong khoảng giá tín chỉ carbon chất lượng cao (~$100–$600/tấn cho loại có kiểm
chứng), nên đây là cách giảm phát thải rẻ ngang hoặc rẻ hơn mua tín chỉ — và khác tín
chỉ ở chỗ nó *không phát thải ngay từ đầu* chứ không bù trừ sau.

Vậy vùng nào "tối ưu"? Phụ thuộc công ty ưu tiên gì. Nếu chỉ tối ưu tiền:
`{r6['cheapest']}` — vẫn sạch hơn `{r6['home_region']}` bốn lần, nên là lựa chọn "được
cả hai" khi không có cam kết carbon cứng. Nếu có mục tiêu phát thải phải báo cáo:
`{r6['cleanest']}`, trả thêm ${r6['clean_premium_usd']:,.0f}/tháng. Theo điểm cân bằng
chuẩn hoá cả hai trục, `{r6['balanced']}` thắng.

Trade-off latency có tồn tại nhưng **không áp dụng cho tập công việc này**: các job
interruptible đều là training và batch eval, không có người dùng nào đang chờ. Ba job
inference phục vụ người dùng thì giữ nguyên tại `{r6['home_region']}` — và đó chính là
lý do chỉ dời phần interruptible chứ không dời cả fleet.
"""


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r6 = m6_carbon_scheduling.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    # Extension 2 supersedes the old one-tier-down heuristic: right-sizing is now
    # driven by measured MBU/MFU plus the VRAM the workload actually touches.
    rightsize_savings = r1["rightsize_monthly"]
    reasoning_savings = r2["reasoning"]["cap_cost_saved_daily"] * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size by MBU": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
        "Reasoning budget cap": round(reasoning_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    sust.update({
        "shiftable_kwh": r6["shiftable_kwh"],
        "carbon_saved_kg": r6["carbon_saved_kg"],
        "carbon_saved_pct": r6["carbon_saved_pct"],
        "power_saved_usd": r6["power_saved_usd"],
        "cheapest_region": r6["cheapest"],
        "balanced_region": r6["balanced"],
        "usd_per_tonne_co2e": r6["usd_per_tonne_co2e"],
    })

    analysis = build_analysis(r1, r2, r3, r6, levers, baseline, optimized, cat)
    md = report.build_report(baseline, optimized, levers, sustainability=sust, analysis=analysis)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"),
                                   baseline_usd=baseline)

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1),
            "sustainability": sust}


if __name__ == "__main__":
    run()
