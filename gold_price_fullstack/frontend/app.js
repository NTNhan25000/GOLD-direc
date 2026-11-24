/* Frontend dashboard — dự đoán giá vàng GC=F */
"use strict";

const $ = (sel) => document.querySelector(sel);
const fmtUSD = (v, d = 2) =>
  Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

const CHART_COLORS = {
  actual: "#f0b429",
  naive: "#4dabf7",
  linear: "#ffa94d",
  rf: "#3ddc97",
  xgb: "#b197fc",
};

let historyChart, testChart;
let historyData = [];

/* ------------------------------------------------------------------ helpers */
async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

function baseChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { color: "#93a0bb", boxWidth: 12, font: { size: 11 } } },
      tooltip: {
        callbacks: {
          label: (ctx) => ` ${ctx.dataset.label}: ${fmtUSD(ctx.parsed.y)}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#93a0bb", maxRotation: 0, autoSkip: true, maxTicksLimit: 10 },
        grid: { color: "#2a355044" },
      },
      y: {
        ticks: { color: "#93a0bb", callback: (v) => fmtUSD(v, 0) },
        grid: { color: "#2a355066" },
      },
    },
  };
}

/* ------------------------------------------------------------- metric cards */
function renderCards(metrics, last) {
  const naive = metrics.find((m) => m.model === "Naive");
  const bestLearned = metrics
    .filter((m) => m.model !== "Naive")
    .reduce((a, b) => (b.MAE < a.MAE ? b : a));
  const prev = historyData.length > 1 ? historyData[historyData.length - 2] : null;
  const change = prev ? last.Gold_Close - prev.Gold_Close : 0;
  const changePct = prev ? (change / prev.Gold_Close) * 100 : 0;
  const cls = change >= 0 ? "up" : "down";
  const arrow = change >= 0 ? "▲" : "▼";

  $("#metric-cards").innerHTML = `
    <div class="card">
      <div class="label">Giá đóng cửa gần nhất</div>
      <div class="value gold">${fmtUSD(last.Gold_Close)} <span style="font-size:13px;color:var(--muted)">USD/oz</span></div>
      <div class="sub">${last.Date} · <span class="${cls}">${arrow} ${fmtUSD(Math.abs(change))} (${fmtUSD(changePct)}%)</span></div>
    </div>
    <div class="card">
      <div class="label">Naive — sai số Test 2024</div>
      <div class="value">${fmtUSD(naive.MAE)}</div>
      <div class="sub">MAE USD/oz · RMSE ${fmtUSD(naive.RMSE)} · MAPE ${fmtUSD(naive.MAPE_pct)}%</div>
    </div>
    <div class="card">
      <div class="label">Mô hình học tốt nhất (Test)</div>
      <div class="value">${bestLearned.model}</div>
      <div class="sub">MAE ${fmtUSD(bestLearned.MAE)} · RMSE ${fmtUSD(bestLearned.RMSE)} · R² ${fmtUSD(bestLearned.R2, 4)}</div>
    </div>
    <div class="card">
      <div class="label">Số phiên Test / tổng</div>
      <div class="value">252 <span style="font-size:14px;color:var(--muted)">/ 2.514</span></div>
      <div class="sub">Train 1.739 · Val 502 · Test 252 (2024)</div>
    </div>`;
}

/* ----------------------------------------------------------- metrics table */
function renderMetricsTable(metrics) {
  const bestMAE = Math.min(...metrics.filter((m) => m.model !== "Naive").map((m) => m.MAE));
  const naiveMAE = metrics.find((m) => m.model === "Naive").MAE;
  const tbody = $("#metrics-table tbody");
  tbody.innerHTML = metrics
    .map((m) => {
      const isNaive = m.model === "Naive";
      const best = !isNaive && m.MAE === bestMAE;
      const bestOverall = isNaive && naiveMAE <= bestMAE;
      const cls = best || bestOverall ? 'class="best-row"' : "";
      const k = m.K == null || Number.isNaN(m.K) ? "—" : m.K;
      return `<tr ${cls}>
        <td>${m.model}${bestOverall ? " (mốc cơ sở)" : ""}</td>
        <td>${k}</td>
        <td>${fmtUSD(m.MAE)}</td>
        <td>${fmtUSD(m.RMSE)}</td>
        <td>${fmtUSD(m.MAPE_pct)}%</td>
        <td>${fmtUSD(m.R2, 4)}</td>
      </tr>`;
    })
    .join("");
}

/* ----------------------------------------------------------- history chart */
const RANGE_MAP = {
  all: ["2015-01-01", "2024-12-31"],
  train: ["2015-01-01", "2021-12-31"],
  val: ["2022-01-01", "2023-12-31"],
  test: ["2024-01-01", "2024-12-31"],
};

function drawHistory(range = "all") {
  const [s, e] = RANGE_MAP[range];
  const rows = historyData.filter((r) => r.Date >= s && r.Date <= e);
  const cfg = {
    type: "line",
    data: {
      labels: rows.map((r) => r.Date),
      datasets: [
        {
          label: "Gold Close (GC=F)",
          data: rows.map((r) => r.Gold_Close),
          borderColor: CHART_COLORS.actual,
          backgroundColor: "#f0b42918",
          borderWidth: 1.8,
          fill: true,
          pointRadius: 0,
          tension: 0.15,
        },
      ],
    },
    options: baseChartOptions(),
  };
  if (historyChart) historyChart.destroy();
  historyChart = new Chart($("#historyChart"), cfg);
}

/* ------------------------------------------------------------- test chart */
function drawTest(preds) {
  const cfg = {
    type: "line",
    data: {
      labels: preds.map((p) => p.Date),
      datasets: [
        mkLine("Thực tế", preds, "Actual", CHART_COLORS.actual, 2),
        mkLine("Naive", preds, "Naive", CHART_COLORS.naive, 1.2),
        mkLine("Linear Regression", preds, "LinearRegression", CHART_COLORS.linear, 1.2),
        mkLine("Random Forest", preds, "RandomForest", CHART_COLORS.rf, 1.2),
        mkLine("XGBoost", preds, "XGBoost", CHART_COLORS.xgb, 1.2),
      ].filter(Boolean),
    },
    options: baseChartOptions(),
  };
  if (testChart) testChart.destroy();
  testChart = new Chart($("#testChart"), cfg);
}

function mkLine(label, rows, key, color, width) {
  if (!rows.length || rows[0][key] === undefined) return null;
  return {
    label,
    data: rows.map((r) => r[key]),
    borderColor: color,
    borderWidth: width,
    pointRadius: 0,
    tension: 0.1,
  };
}

/* --------------------------------------------------------------- forecast */
function renderForecast(last, forecast, hint) {
  $("#forecast-hint").textContent = hint || "";
  const entries = Object.entries(forecast);
  const base = forecast.Naive;
  // "best" = mô hình học có MAE test thấp nhất trong bảng metrics
  $("#forecast-grid").innerHTML = entries
    .map(([name, val]) => {
      const delta = val - base;
      const dCls = delta >= 0 ? "up" : "down";
      const dTxt =
        name === "Naive"
          ? "mốc cơ sở"
          : `${delta >= 0 ? "+" : ""}${fmtUSD(delta)} so với Naive`;
      return `<div class="forecast-item">
        <div class="fname">${name}</div>
        <div class="fval" style="color:${CHART_COLORS[nameKey(name)] || "var(--text)"}">${fmtUSD(val)}</div>
        <div class="fdelta ${dCls}">${dTxt}</div>
        ${name === "Naive" ? '<span class="tag">Baseline</span>' : ""}
      </div>`;
    })
    .join("");
}

function nameKey(n) {
  return { Naive: "naive", LinearRegression: "linear", RandomForest: "rf", XGBoost: "xgb" }[n] || "actual";
}

/* ------------------------------------------------------- manual prediction */
function fillManualForm(sess) {
  $("#f-date").value = sess.date;
  $("#f-close").value = sess.Gold_Close.toFixed(2);
  $("#f-open").value = sess.Gold_Open.toFixed(2);
  $("#f-high").value = sess.Gold_High.toFixed(2);
  $("#f-low").value = sess.Gold_Low.toFixed(2);
  $("#f-dxy").value = Number(sess.DXY_Close).toFixed(3);
  $("#f-teny").value = Number(sess.US10Y_Close).toFixed(3);
  $("#f-spx").value = Number(sess.SP500_Close).toFixed(2);
  $("#f-wti").value = Number(sess.WTI_Close).toFixed(2);
  $("#f-silver").value = Number(sess.Silver_Close).toFixed(2);
}

async function onManualSubmit(ev) {
  ev.preventDefault();
  const box = $("#manual-result");
  box.classList.remove("hidden");
  box.innerHTML = "<p>Đang dự báo…</p>";
  const fd = new FormData($("#manual-form"));
  const body = {};
  for (const [k, v] of fd.entries()) {
    if (v !== "" && v != null) body[k] = Number.isNaN(Number(v)) ? v : Number(v);
  }
  try {
    const res = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const rows = Object.entries(res.forecast_next_session)
      .map(([k, v]) => `<li><b>${k}:</b> ${fmtUSD(v)} USD/oz</li>`)
      .join("");
    box.innerHTML = `
      <h3>Dự báo cho phiên sau ${res.origin_date}</h3>
      <ul>${rows}</ul>
      <p class="note" style="margin-top:8px">${res.note}</p>`;
  } catch (err) {
    box.innerHTML = `<p class="err">Lỗi: ${err.message}</p>`;
  }
}

/* ------------------------------------------------------------------- init */
async function init() {
  try {
    const [history, metrics, preds, last] = await Promise.all([
      api("/api/history"),
      api("/api/metrics"),
      api("/api/predictions"),
      api("/api/last-session"),
    ]);

    historyData = history;
    renderCards(metrics, last.session);
    renderMetricsTable(metrics);
    drawHistory("all");
    drawTest(preds);

    const f = last.forecast_next_session;
    renderForecast(last.session, f, last.target_date_hint);
    fillManualForm(last.session);
    $("#last-session-pill").textContent =
      `Phiên gần nhất: ${last.session.date} · ${fmtUSD(last.session.Gold_Close)} USD/oz`;

    document.querySelectorAll("#history-tabs .tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#history-tabs .tab").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        drawHistory(btn.dataset.range);
      });
    });

    $("#manual-form").addEventListener("submit", onManualSubmit);
    $("#btn-reset").addEventListener("click", () => fillManualForm(last.session));
  } catch (err) {
    $("#metric-cards").innerHTML =
      `<div class="card" style="grid-column:1/-1"><div class="value" style="color:var(--red)">Không tải được dữ liệu</div>
       <div class="sub">${err.message} — hãy chắc chắn backend đang chạy và đã huấn luyện mô hình (<code>python ml/train.py</code>).</div></div>`;
  }
}

init();
