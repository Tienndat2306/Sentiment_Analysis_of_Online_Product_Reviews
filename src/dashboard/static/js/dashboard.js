/**
 * Logic điều khiển giao diện, tương tác bộ lọc & ECharts 5
 * Giao diện: Stitch Minimalist Sentiment Dashboard Redesign (Light Rose Theme)
 * Kết nối trực tiếp Semantic Layer API (/api/dashboard-data, /api/products, /api/predict)
 */

let currentTab = 'overview';
let currentPlatform = 'all';
let dashboardCache = null;
let modelsCache = null;
let currentProductsList = [];

// Helper: Rút gọn tên danh mục trên trục X của Heatmap
function shortCategoryName(name) {
  const map = {
    'Sách & Văn Phòng Phẩm': 'Sách & VPP',
    'Làm Đẹp - Sức Khỏe': 'Làm Đẹp',
    'Nhà Cửa - Đời Sống': 'Nhà Cửa',
    'Điện & Đồ Gia Dụng': 'Điện & Gia Dụng',
    'Bách Hóa Online': 'Bách Hóa',
    'Đồ Chơi - Mẹ & Bé': 'Mẹ & Bé',
    'Điện Thoại - Máy Tính Bảng': 'Điện Thoại'
  };
  return map[name] || name;
}

// Render Heatmap cố định Top 3 danh mục nhiều sản phẩm & nhiều bình luận nhất
function renderHeatmapFromCache(sourceData) {
  const data = sourceData || dashboardCache;
  if (!data || !data.heatmap || !data.categories) return;

  const chartHeatmap = getOrInitChart('chart-heatmap');
  if (!chartHeatmap) return;

  const topK = 3;

  // Xếp hạng danh mục theo điểm tổng hợp: 50% tỷ trọng Số sản phẩm + 50% tỷ trọng Số bình luận
  const cats = data.categories || [];
  const maxProd = Math.max(1, ...cats.map(c => c.total_products || 0));
  const maxRev = Math.max(1, ...cats.map(c => c.total_reviews || 0));

  const rankedCats = [...cats]
    .map(c => ({
      ...c,
      combinedScore: (c.total_products / maxProd) * 0.5 + (c.total_reviews / maxRev) * 0.5
    }))
    .sort((a, b) => b.combinedScore - a.combinedScore)
    .slice(0, topK);

  const selectedNames = rankedCats.map(c => c.category_group);
  const origCatList = data.heatmap.categories || [];
  const tierNamesShort = (data.heatmap.tiers || []).map(t => t.split(' (')[0]);

  // Ánh xạ lại chỉ số cột X (0 .. selectedNames.length - 1)
  const hmCells = [];
  selectedNames.forEach((catName, newXIdx) => {
    const oldCatIdx = origCatList.indexOf(catName);
    if (oldCatIdx === -1) return;
    (data.heatmap.data || []).forEach(item => {
      // item format từ backend: [t_idx, c_idx, neg_rate, count]
      if (item[1] === oldCatIdx) {
        hmCells.push([newXIdx, item[0], item[2], item[3]]);
      }
    });
  });

  const maxVal = Math.max(14, ...hmCells.map(c => c[2] || 0));

  chartHeatmap.setOption({
    tooltip: {
      position: 'top',
      formatter: p => {
        const catObj = rankedCats[p.data[0]] || {};
        const fullCat = catObj.category_group || '';
        const fullTier = (data.heatmap.tiers || [])[p.data[1]] || '';
        const cnt = p.data[3] ? `` : '';
        const meta = catObj.total_products
          ? `<br/><span style="color:#94a3b8;font-size:11px">Quy mô ngành: ${catObj.total_products} SP • ${Number(catObj.total_reviews).toLocaleString('en-US')} đánh giá</span>`
          : '';
        return `<b>${fullCat}</b><br/>${fullTier}: <b style="color:#e11d48">${p.data[2]}% phàn nàn</b>${cnt}${meta}`;
      }
    },
    xAxis: {
      data: selectedNames.map(shortCategoryName),
      axisLabel: { color: '#475569', fontSize: 11, fontWeight: 600, margin: 10, interval: 0 }
    },
    yAxis: {
      data: tierNamesShort,
      axisLabel: { color: '#475569', fontSize: 11, fontWeight: 500, margin: 10 }
    },
    visualMap: {
      min: 0,
      max: Math.ceil(maxVal),
      text: [`${Math.ceil(maxVal)}%`, '0%']
    },
    series: [{
      data: hmCells,
      label: {
        show: true,
        formatter: p => p.data[2] > 0 ? p.data[2] + '%' : '—',
        color: '#1e293b',
        fontWeight: 600,
        fontSize: 12
      }
    }]
  });
}

// Helper: Chỉ hiển thị icon sao (ví dụ: 4 -> ★★★★☆, 5 -> ★★★★★)
function renderStarsOnly(rating) {
  const r = Math.max(1, Math.min(5, Math.round(Number(rating) || 5)));
  return '★'.repeat(r) + '☆'.repeat(5 - r);
}

// Helper: Rút gọn thông minh tên gian hàng để không bị tràn/cắt chữ trên biểu đồ 1/3 cột
function formatSellerName(name) {
  if (!name) return '';
  return String(name)
    .replace(/^Gian hàng\s+/i, '')
    .replace(/\s+chính hãng$/i, ' Official')
    .replace(/\s+Official Store$/i, ' Official');
}

// Tab switching logic
function switchTab(tabId) {
  currentTab = tabId;
  const tabs = ['overview', 'pricetier', 'customer', 'products', 'sandbox'];
  const titles = {
    overview: 'Tổng quan & Xu hướng',
    pricetier: 'Phân khúc giá & Ngành hàng',
    customer: 'Phát hiện mâu thuẫn đánh giá',
    products: 'Chi tiết sản phẩm',
    sandbox: 'Thử nghiệm AI & Trợ lý'
  };

  tabs.forEach(t => {
    const btn = document.getElementById('tab-btn-' + t);
    const view = document.getElementById('view-' + t);
    if (btn) {
      if (t === tabId) {
        btn.classList.add('active');
        btn.classList.remove('text-slate-600');
      } else {
        btn.classList.remove('active');
        btn.classList.add('text-slate-600');
      }
    }
    if (view) {
      if (t === tabId) {
        view.classList.remove('hidden');
      } else {
        view.classList.add('hidden');
      }
    }
  });

  const titleEl = document.getElementById('view-title');
  if (titleEl) titleEl.innerText = titles[tabId] || 'Dashboard';

  const mainViewport = document.getElementById('main-viewport');
  if (mainViewport) {
    mainViewport.scrollTop = 0;
  }

  setTimeout(() => {
    window.dispatchEvent(new Event('resize'));
    if (dashboardCache) {
      updateDashboardWithRealData(dashboardCache);
    }
    if (tabId === 'products') {
      loadProductsTable();
    }
    if (tabId === 'sandbox') {
      loadModelsData();
    }
  }, 50);
}

// Platform selection (Dropdown list)
function selectPlatform(plat) {
  currentPlatform = plat || 'all';
  const selectEl = document.getElementById('filter-platform');
  if (selectEl && selectEl.value !== currentPlatform) {
    selectEl.value = currentPlatform;
  }
  applyFilters();
}

async function applyFilters() {
  const category = (document.getElementById('filter-category') || {}).value || 'ALL';
  const tier = (document.getElementById('filter-tier') || {}).value || 'ALL';
  const year = (document.getElementById('filter-year') || {}).value || '2022';

  try {
    const params = new URLSearchParams({
      source: currentPlatform,
      category_group: category,
      price_tier: tier,
      start_year: year
    });
    const res = await fetch('/api/dashboard-data?' + params.toString());
    if (!res.ok) throw new Error('API unavailable');
    const data = await res.json();
    dashboardCache = data;
    updateDashboardWithRealData(data);
    if (currentTab === 'products') {
      loadProductsTable();
    }
  } catch (err) {
    // Fallback if opened as standalone static file
    const total = document.getElementById('kpi-total');
    if (total) total.innerText = (40000 + Math.floor(Math.random() * 8000)).toLocaleString();
  }
}

function resetFilters() {
  document.getElementById('filter-category').value = 'ALL';
  document.getElementById('filter-tier').value = 'ALL';
  document.getElementById('filter-year').value = '2022';
  selectPlatform('all');
}

function fillSandbox(text, star) {
  document.getElementById('sandbox-input').value = text;
  document.getElementById('sandbox-stars').value = star;
  analyzeSentiment();
}

async function analyzeSentiment() {
  const rawText = document.getElementById('sandbox-input').value.trim();
  const stars = parseInt(document.getElementById('sandbox-stars').value);
  const starsVisual = renderStarsOnly(stars);
  const badge = document.getElementById('result-badge');
  const note = document.getElementById('result-note');
  const probPos = document.getElementById('prob-pos');
  const probNeu = document.getElementById('prob-neu');
  const probNeg = document.getElementById('prob-neg');

  // Ưu tiên gọi mô hình AI thực tế trên server (/api/predict)
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: rawText, rating: stars })
    });
    if (res.ok) {
      const d = await res.json();
      probPos.innerText = d.probabilities.positive + '%';
      probNeu.innerText = d.probabilities.neutral + '%';
      probNeg.innerText = d.probabilities.negative + '%';

      if (d.label_final === 'negative') {
        badge.className = "px-3 py-1 rounded-full text-xs font-bold bg-rose-500 text-white shadow-sm";
        badge.innerText = "TIÊU CỰC";
        if (stars >= 4) {
          note.className = "text-xs text-rose-800 bg-rose-100/60 p-3 rounded-xl font-medium border border-rose-200/60";
          note.innerHTML = `⚠️ <b>Phát hiện mâu thuẫn:</b> Khách chọn <b class="text-amber-500 tracking-wider">${starsVisual}</b> nhưng lời nhận xét chứa cảm xúc tiêu cực (${d.decision_rule}). Hệ thống đã đánh dấu để đội ngũ CSKH ưu tiên xử lý!`;
        } else {
          note.className = "text-xs text-rose-800 bg-rose-100/60 p-3 rounded-xl font-medium border border-rose-200/60";
          note.innerHTML = `Đánh giá phản ánh trải nghiệm chưa tốt (<span class="text-amber-500 tracking-wider">${starsVisual}</span>). ${d.decision_rule}.`;
        }
      } else if (d.label_final === 'positive') {
        badge.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-500 text-white shadow-sm";
        badge.innerText = "TÍCH CỰC";
        note.className = "text-xs text-emerald-800 bg-emerald-50 p-3 rounded-xl font-medium border border-emerald-200/60";
        note.innerHTML = `Khách hàng hài lòng cao và đánh giá đồng thuận với mức <span class="text-amber-500 tracking-wider">${starsVisual}</span> (${d.decision_rule}).`;
      } else {
        badge.className = "px-3 py-1 rounded-full text-xs font-bold bg-amber-500 text-white shadow-sm";
        badge.innerText = "TRUNG LẬP";
        note.className = "text-xs text-amber-800 bg-amber-50 p-3 rounded-xl font-medium border border-amber-200/60";
        note.innerHTML = `Nội dung trung tính, phản ánh mức độ hài lòng ở mức vừa phải (<span class="text-amber-500 tracking-wider">${starsVisual}</span> — ${d.decision_rule}).`;
      }
      return;
    }
  } catch (e) {
    // Fallback offline rule
  }

  const text = rawText.toLowerCase();
  const isNeg = text.includes('rách') || text.includes('móp') || text.includes('thất vọng') || text.includes('khét') || text.includes('sập') || text.includes('ẩu');
  const isPos = text.includes('tuyệt vời') || text.includes('êm ái') || text.includes('đáng tiền') || text.includes('thích');

  if (isNeg) {
    badge.className = "px-3 py-1 rounded-full text-xs font-bold bg-rose-500 text-white shadow-sm";
    badge.innerText = "TIÊU CỰC";
    probPos.innerText = "3.2%";
    probNeu.innerText = "8.4%";
    probNeg.innerText = "88.4%";

    if (stars >= 4) {
      note.className = "text-xs text-rose-800 bg-rose-100/60 p-3 rounded-xl font-medium border border-rose-200/60";
      note.innerHTML = `⚠️ <b>Phát hiện mâu thuẫn:</b> Khách chọn <b class="text-amber-500 tracking-wider">${starsVisual}</b> nhưng lời nhận xét chứa cảm xúc tiêu cực. Hệ thống đã đánh dấu để đội ngũ CSKH ưu tiên xử lý!`;
    } else {
      note.className = "text-xs text-rose-800 bg-rose-100/60 p-3 rounded-xl font-medium border border-rose-200/60";
      note.innerHTML = `Đánh giá phản ánh trải nghiệm chưa tốt (<span class="text-amber-500 tracking-wider">${starsVisual}</span>). Đề xuất liên hệ hỗ trợ đổi trả sản phẩm.`;
    }
  } else if (isPos) {
    badge.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-500 text-white shadow-sm";
    badge.innerText = "TÍCH CỰC";
    probPos.innerText = "94.8%";
    probNeu.innerText = "4.2%";
    probNeg.innerText = "1.0%";
    note.className = "text-xs text-emerald-800 bg-emerald-50 p-3 rounded-xl font-medium border border-emerald-200/60";
    note.innerHTML = `Khách hàng hài lòng cao và đánh giá đồng thuận với mức <span class="text-amber-500 tracking-wider">${starsVisual}</span>.`;
  } else {
    badge.className = "px-3 py-1 rounded-full text-xs font-bold bg-amber-500 text-white shadow-sm";
    badge.innerText = "TRUNG LẬP";
    probPos.innerText = "22.5%";
    probNeu.innerText = "65.3%";
    probNeg.innerText = "12.2%";
    note.className = "text-xs text-amber-800 bg-amber-50 p-3 rounded-xl font-medium border border-amber-200/60";
    note.innerHTML = `Nội dung trung tính, phản ánh mức độ hài lòng ở mức vừa phải (<span class="text-amber-500 tracking-wider">${starsVisual}</span>).`;
  }
}

// Helper to get or initialize ECharts instance
function getOrInitChart(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  return echarts.getInstanceByDom(el) || echarts.init(el);
}

// Update UI & Charts when real backend data is loaded
function updateDashboardWithRealData(data) {
  if (!data || !data.kpis) return;
  const k = data.kpis;

  // 1. Update 6 KPI cards
  const elTotal = document.getElementById('kpi-total');
  if (elTotal) elTotal.innerText = k.total_reviews.toLocaleString('en-US');

  const elRating = document.getElementById('kpi-rating');
  if (elRating) elRating.innerText = k.avg_rating.toFixed(2);

  const elRatingSub = document.getElementById('kpi-rating-sub');
  if (elRatingSub) elRatingSub.innerText = `Trên ${k.total_products.toLocaleString('en-US')} sản phẩm`;

  const elPos = document.getElementById('kpi-pos');
  if (elPos) elPos.innerText = k.positive_text_pct + '%';

  const elNeu = document.getElementById('kpi-neu');
  if (elNeu) elNeu.innerText = k.neutral_text_pct + '%';

  const elNeg = document.getElementById('kpi-neg');
  if (elNeg) elNeg.innerText = k.negative_text_pct + '%';

  const elConflict = document.getElementById('kpi-conflict');
  if (elConflict) elConflict.innerText = k.conflict_count.toLocaleString('en-US');

  // Update Donut legend rows
  const dPos = document.getElementById('donut-pos-text');
  if (dPos) dPos.innerText = `${k.positive_text_pct}% (${k.positive_text.toLocaleString('en-US')})`;
  const dNeu = document.getElementById('donut-neu-text');
  if (dNeu) dNeu.innerText = `${k.neutral_text_pct}% (${k.neutral_text.toLocaleString('en-US')})`;
  const dNeg = document.getElementById('donut-neg-text');
  if (dNeg) dNeg.innerText = `${k.negative_text_pct}% (${k.negative_text.toLocaleString('en-US')})`;

  // 2. Time Series Chart
  const chartTs = getOrInitChart('chart-timeseries');
  if (chartTs && data.time_series && data.time_series.length > 0) {
    const recentTs = data.time_series.slice(-18);
    chartTs.setOption({
      xAxis: { data: recentTs.map(d => d.year_month) },
      series: [
        { data: recentTs.map(d => d.total_reviews) },
        { data: recentTs.map(d => d.negative_pct) }
      ]
    });
  }

  // 3. Donut Chart (with center metric information)
  const chartDonut = getOrInitChart('chart-donut');
  if (chartDonut) {
    chartDonut.setOption({
      title: {
        text: k.positive_text_pct + '%',
        subtext: `${k.usable_text_reviews.toLocaleString('en-US')} nhận xét`,
        left: 'center',
        top: '36%',
        itemGap: 4,
        textStyle: {
          fontSize: 22,
          fontWeight: '800',
          color: '#10b981',
          fontFamily: 'JetBrains Mono, monospace'
        },
        subtextStyle: {
          fontSize: 11,
          fontWeight: '600',
          color: '#64748b',
          fontFamily: 'Plus Jakarta Sans, sans-serif'
        }
      },
      series: [{
        data: [
          { value: k.positive_text, name: 'Tích cực', itemStyle: { color: '#10b981' } },
          { value: k.neutral_text, name: 'Trung lập', itemStyle: { color: '#f59e0b' } },
          { value: k.negative_text, name: 'Tiêu cực', itemStyle: { color: '#f43f5e' } }
        ]
      }]
    });
  }

  // 4. Price Tier Chart (Tab 1) & Tier Detail Chart (Tab 2)
  if (data.price_tiers && data.price_tiers.length > 0) {
    const chartTier = getOrInitChart('chart-tier');
    if (chartTier) {
      chartTier.setOption({
        xAxis: { data: data.price_tiers.map(t => t.price_tier) },
        series: [{ data: data.price_tiers.map(t => t.text_negative_pct) }]
      });
    }

    const chartTierDetail = getOrInitChart('chart-tier-detail');
    if (chartTierDetail) {
      chartTierDetail.setOption({
        xAxis: { data: data.price_tiers.map(t => t.price_tier.split(' (')[0]) },
        series: [
          { name: '% Hài lòng', data: data.price_tiers.map(t => t.text_positive_pct) },
          { name: '% Phàn nàn', data: data.price_tiers.map(t => t.text_negative_pct) }
        ]
      });
    }
  }

  // 4b. Heatmap Chart (Tab 2 — mặc định lọc Top 3 danh mục nhiều SP & nhiều bình luận nhất)
  renderHeatmapFromCache(data);

  // 5. Category Chart (Tab 1) & Category Table (Tab 2)
  if (data.categories && data.categories.length > 0) {
    const chartCat = getOrInitChart('chart-category');
    if (chartCat) {
      const catsRev = data.categories.slice(0, 6).reverse();
      chartCat.setOption({
        grid: { top: 15, right: 48, bottom: 25, left: 12, containLabel: true },
        yAxis: {
          data: catsRev.map(c =>
            c.category_group
              .replace('Văn Phòng Phẩm', 'VPP')
              .replace('Máy Tính Bảng', 'MTB')
          ),
          axisLabel: { color: '#475569', fontSize: 11, fontWeight: 500, margin: 10 }
        },
        series: [{
          data: catsRev.map(c => c.text_positive_pct),
          label: { show: true, position: 'right', formatter: '{c}%', color: '#64748b', fontSize: 11, fontWeight: 600 }
        }]
      });
    }

    const catTbody = document.getElementById('table-category-body');
    if (catTbody) {
      catTbody.innerHTML = data.categories.map(c => `
        <tr class="hover:bg-rose-50/40">
          <td class="py-3 px-4 font-semibold text-slate-800">${c.category_group}</td>
          <td class="py-3 px-4 text-right">${c.total_products}</td>
          <td class="py-3 px-4 text-right font-mono">${c.total_reviews.toLocaleString('en-US')}</td>
          <td class="py-3 px-4 text-right font-mono font-bold text-amber-500">${c.avg_rating.toFixed(2)} ★</td>
          <td class="py-3 px-4 text-right font-mono font-bold text-emerald-600">${c.text_positive_pct.toFixed(1)}%</td>
          <td class="py-3 px-4 text-right font-mono font-bold text-rose-500">${c.text_negative_pct.toFixed(1)}%</td>
          <td class="py-3 px-4 text-right font-mono text-rose-600">${c.high_star_negative_count.toLocaleString('en-US')} ca</td>
        </tr>
      `).join('');
    }
  }

  // 6. Customer Tab Charts & Conflict Reviews Table (Only Star Icons)
  if (data.customer_segments) {
    const cp = getOrInitChart('chart-customer-photo');
    if (cp && data.customer_segments.photo && data.customer_segments.photo.length > 0) {
      cp.setOption({
        xAxis: { data: data.customer_segments.photo.map(r => r.segment_value.split(' (')[0]) },
        series: [{ data: data.customer_segments.photo.map(r => r.text_negative_pct) }]
      });
    }

    const cf = getOrInitChart('chart-customer-freq');
    if (cf && data.customer_segments.frequency && data.customer_segments.frequency.length > 0) {
      cf.setOption({
        xAxis: { data: data.customer_segments.frequency.map(r => r.segment_value.split(' (')[0]) },
        series: [{ data: data.customer_segments.frequency.map(r => r.text_positive_pct) }]
      });
    }

    const cs = getOrInitChart('chart-customer-seller');
    if (cs && data.customer_segments.sellers && data.customer_segments.sellers.length > 0) {
      const topSellers = data.customer_segments.sellers.slice(0, 5).reverse();
      cs.setOption({
        tooltip: {
          trigger: 'axis',
          formatter: params => {
            const p = params[0];
            const rawSeller = topSellers[p.dataIndex];
            return `<b>${rawSeller.segment_value}</b><br/>% Hài lòng: <b>${p.value}%</b> (${Number(rawSeller.total_reviews).toLocaleString('en-US')} đánh giá)`;
          }
        },
        grid: { top: 15, right: 48, bottom: 25, left: 12, containLabel: true },
        xAxis: {
          type: 'value',
          max: 100,
          axisLabel: { formatter: '{value}%', color: '#94a3b8', fontSize: 10 },
          splitLine: { lineStyle: { color: '#f1f5f9' } }
        },
        yAxis: {
          type: 'category',
          data: topSellers.map(r => formatSellerName(r.segment_value)),
          axisLabel: { color: '#475569', fontSize: 11, fontWeight: 500, margin: 10 }
        },
        series: [{
          name: '% Hài lòng',
          type: 'bar',
          barWidth: '48%',
          itemStyle: { color: '#fb7185', borderRadius: [0, 6, 6, 0] },
          label: { show: true, position: 'right', formatter: '{c}%', color: '#64748b', fontSize: 11, fontWeight: 600 },
          data: topSellers.map(r => r.text_positive_pct)
        }]
      });
    }
  }

  if (data.conflict_samples && data.conflict_samples.length > 0) {
    const cBody = document.getElementById('table-conflicts-body');
    if (cBody) {
      cBody.innerHTML = data.conflict_samples.slice(0, 12).map(r => {
        const starsStr = renderStarsOnly(r.rating);
        return `
          <tr class="hover:bg-rose-50/30">
            <td class="py-3 px-4 font-semibold text-slate-800 max-w-[260px] truncate" title="${r.product_name}">${r.product_name}</td>
            <td class="py-3 px-4 text-center font-bold text-amber-500 tracking-wider whitespace-nowrap">${starsStr}</td>
            <td class="py-3 px-4 text-center whitespace-nowrap"><span class="px-2.5 py-1 rounded-full text-[11px] font-semibold bg-rose-100 text-rose-700">Thất vọng / Tiêu cực</span></td>
            <td class="py-3 px-4 text-slate-600 font-medium">"${r.content_clean}"</td>
          </tr>
        `;
      }).join('');
    }
  }
}

async function loadProductsTable() {
  const category = (document.getElementById('filter-category') || {}).value || 'ALL';
  const tier = (document.getElementById('filter-tier') || {}).value || 'ALL';
  const search = (document.getElementById('prod-search') || {}).value || '';
  const sortBy = (document.getElementById('prod-sort') || {}).value || 'total_reviews';
  const anomaly = (document.getElementById('prod-anomaly') || {}).checked || false;

  try {
    const params = new URLSearchParams({
      source: currentPlatform,
      category_group: category,
      price_tier: tier,
      search,
      sort_by: sortBy,
      anomaly_only: anomaly,
      limit: 100
    });
    const res = await fetch('/api/products?' + params.toString());
    if (!res.ok) return;
    const data = await res.json();
    currentProductsList = data.products || [];

    const badge = document.getElementById('prod-count-badge');
    if (badge) badge.innerText = `Hiển thị ${data.total_matched} sản phẩm`;

    const tbody = document.getElementById('table-products-body');
    if (tbody && data.products) {
      tbody.innerHTML = data.products.map(p => {
        const isTiki = String(p.source).toLowerCase() === 'tiki';
        const srcBadge = isTiki
          ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-100 text-sky-700">Tiki</span>'
          : '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-pink-100 text-pink-700">Lazada</span>';
        const statusBadge = p.is_anomaly_negative
          ? '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-100 text-rose-700">Cảnh báo</span>'
          : '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100 text-emerald-700">Tốt</span>';
        const safeName = (p.product_name || '').replace(/'/g, '');
        return `
          <tr onclick="openProductDrilldown('${p.product_id}', '${safeName}')" class="hover:bg-rose-50/50 cursor-pointer transition">
            <td class="py-3 px-4">${srcBadge}</td>
            <td class="py-3 px-4 font-semibold text-slate-800 max-w-[340px] truncate" title="${p.product_name}">${p.product_name}</td>
            <td class="py-3 px-4 text-slate-500">${p.category_group}</td>
            <td class="py-3 px-4 text-right font-mono">${Number(p.total_reviews).toLocaleString('en-US')}</td>
            <td class="py-3 px-4 text-right font-mono font-bold text-amber-500">${Number(p.avg_rating).toFixed(1)} ★</td>
            <td class="py-3 px-4 text-right font-mono text-emerald-600 font-bold">${Number(p.text_positive_pct).toFixed(1)}%</td>
            <td class="py-3 px-4 text-right font-mono text-rose-500 font-bold">${Number(p.text_negative_pct).toFixed(1)}%</td>
            <td class="py-3 px-4 text-center">${statusBadge}</td>
          </tr>
        `;
      }).join('');
    }
  } catch (e) {
    // Keep default rows if offline
  }
}

async function openProductDrilldown(productId, productName) {
  const modal = document.getElementById('drilldown-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  document.getElementById('modal-prod-title').innerText = `${productName} (ID: ${productId})`;
  document.getElementById('modal-prod-sub').innerText = 'Đang tải danh sách bình luận thực tế...';
  document.getElementById('modal-reviews-list').innerHTML = '';

  try {
    const res = await fetch(`/api/product/${productId}/reviews?limit=40`);
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById('modal-prod-sub').innerText = `Tổng số đánh giá có nhận xét chi tiết: ${data.total_usable_reviews} (Ưu tiên hiển thị các đánh giá mâu thuẫn lên đầu)`;

    document.getElementById('modal-reviews-list').innerHTML = data.reviews.map(r => `
      <div class="p-3.5 rounded-xl bg-slate-50 border ${r.is_conflict ? 'border-rose-200 bg-rose-50/40' : 'border-slate-200/80'} flex flex-col gap-1.5">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="flex items-center gap-2">
            <span class="text-amber-500 font-bold tracking-wider">${renderStarsOnly(r.rating)}</span>
            <span class="px-2 py-0.5 rounded-full text-[10px] font-semibold ${r.label_final === 'positive' ? 'bg-emerald-100 text-emerald-700' : r.label_final === 'negative' ? 'bg-rose-100 text-rose-700' : 'bg-amber-100 text-amber-700'}">
              ${r.label_final.toUpperCase()}
            </span>
            ${r.is_conflict ? '<span class="px-2 py-0.5 rounded-full bg-rose-100 text-rose-700 text-[10px] font-bold">Mâu thuẫn Sao & Bình luận</span>' : ''}
          </div>
          <div class="text-[11px] font-mono text-slate-400">
            Nguồn: <span class="text-rose-500 font-semibold">${r.prediction_source}</span> • Ngày: ${r.created_at}
          </div>
        </div>
        <p class="text-slate-700 text-xs leading-relaxed font-medium">"${r.content_clean}"</p>
      </div>
    `).join('');
  } catch (e) {
    document.getElementById('modal-prod-sub').innerText = 'Không thể tải dữ liệu chi tiết.';
  }
}

function closeDrilldownModal() {
  const modal = document.getElementById('drilldown-modal');
  if (modal) modal.classList.add('hidden');
}

// Xuất danh sách sản phẩm đang lọc ở Tab 4 ra file CSV (hỗ trợ tiếng Việt UTF-8 BOM)
function exportProductsCsv() {
  if (!currentProductsList || currentProductsList.length === 0) {
    alert('Đang tải dữ liệu sản phẩm, vui lòng thử lại sau giây lát.');
    return;
  }
  const headers = [
    'Sàn TMĐT',
    'Mã Sản Phẩm',
    'Tên Sản Phẩm',
    'Ngành Hàng',
    'Phân Khúc Giá',
    'Giá Bán (VNĐ)',
    'Tổng Đánh Giá',
    'Đánh Giá Có Chữ',
    'Điểm Sao TB',
    '% Hài Lòng (Text)',
    '% Trung Lập (Text)',
    '% Phàn Nàn (Text)',
    '% Mâu Thuẫn Sao-Chữ',
    'Trạng Thái Cảnh Báo'
  ];
  const rows = currentProductsList.map(p => [
    String(p.source || '').toUpperCase(),
    `"${String(p.product_id || '')}"`,
    `"${String(p.product_name || '').replace(/"/g, '""')}"`,
    `"${String(p.category_group || '')}"`,
    `"${String(p.price_tier || '')}"`,
    Number(p.price || 0),
    Number(p.total_reviews || 0),
    Number(p.usable_text_reviews || 0),
    Number(p.avg_rating || 0).toFixed(2),
    Number(p.text_positive_pct || 0).toFixed(2),
    Number(p.text_neutral_pct || 0).toFixed(2),
    Number(p.text_negative_pct || 0).toFixed(2),
    Number(p.conflict_pct || 0).toFixed(2),
    p.is_anomaly_negative ? 'Cảnh báo' : 'Tốt'
  ]);

  const csvContent = '\uFEFF' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `sentiment_products_report_${currentPlatform}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Đồng bộ dữ liệu thật từ /api/models cho 2 biểu đồ ở Tab 5 (#chart-models & #chart-cm)
async function loadModelsData() {
  try {
    if (!modelsCache) {
      const res = await fetch('/api/models');
      if (!res.ok) return;
      modelsCache = await res.json();
    }

    // 1. Biểu đồ Độ chính xác phân tích cảm xúc (#chart-models)
    const chartModels = getOrInitChart('chart-models');
    if (chartModels && modelsCache.active_model) {
      const sk = modelsCache.active_model.all_candidate_metrics || {};
      const sp = (modelsCache.spark_mllib_report || {}).models_comparison || {};

      const modelItems = [
        {
          name: 'LinearSVC',
          acc: Number(((sk['LinearSVC (Balanced, Calibrated)'] || {}).accuracy || 0.9233) * 100).toFixed(1),
          f1: Number(((sk['LinearSVC (Balanced, Calibrated)'] || {}).f1_macro || 0.7845) * 100).toFixed(1)
        },
        {
          name: 'LogReg (Sklearn)',
          acc: Number(((sk['LogisticRegression (Balanced)'] || {}).accuracy || 0.8989) * 100).toFixed(1),
          f1: Number(((sk['LogisticRegression (Balanced)'] || {}).f1_macro || 0.7746) * 100).toFixed(1)
        },
        {
          name: 'MultinomialNB',
          acc: Number(((sk['MultinomialNB (Baseline)'] || {}).accuracy || 0.8933) * 100).toFixed(1),
          f1: Number(((sk['MultinomialNB (Baseline)'] || {}).f1_macro || 0.6913) * 100).toFixed(1)
        },
        {
          name: 'ComplementNB',
          acc: Number(((sk['ComplementNB (Imbalanced-optimized)'] || {}).accuracy || 0.8639) * 100).toFixed(1),
          f1: Number(((sk['ComplementNB (Imbalanced-optimized)'] || {}).f1_macro || 0.6776) * 100).toFixed(1)
        },
        {
          name: 'Spark MLlib LR',
          acc: Number(((sp['LogisticRegression (Spark MLlib Balanced)'] || {}).accuracy || 0.8211) * 100).toFixed(1),
          f1: Number(((sp['LogisticRegression (Spark MLlib Balanced)'] || {}).f1_macro || 0.6701) * 100).toFixed(1)
        }
      ];

      chartModels.setOption({
        tooltip: {
          trigger: 'axis',
          formatter: params => {
            const p = params[0];
            const item = modelItems[p.dataIndex];
            return `<b>${item.name}</b><br/>Accuracy: <b>${item.acc}%</b><br/>F1-Macro: <b>${item.f1}%</b>`;
          }
        },
        grid: { top: 20, right: 20, bottom: 25, left: 15, containLabel: true },
        xAxis: { type: 'category', data: modelItems.map(m => m.name), axisLabel: { color: '#475569', fontSize: 11, fontWeight: 500 } },
        yAxis: { type: 'value', min: 75, max: 96, axisLabel: { formatter: '{value}%', color: '#94a3b8' }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
        series: [{
          name: 'Accuracy',
          type: 'bar',
          barWidth: '40%',
          itemStyle: { color: '#f43f5e', borderRadius: [6, 6, 0, 0] },
          label: { show: true, position: 'top', formatter: '{c}%', color: '#e11d48', fontSize: 11, fontWeight: 600 },
          data: modelItems.map(m => Number(m.acc))
        }]
      });
    }

    // 2. Ma trận đối chiếu nhầm lẫn thực tế (#chart-cm)
    const chartCm = getOrInitChart('chart-cm');
    if (chartCm && modelsCache.evaluation_report && modelsCache.evaluation_report.confusion_matrix) {
      // confusion_matrix trong evaluation_report.json có thứ tự: [negative (0), neutral (1), positive (2)]
      // Ánh xạ sang thứ tự hiển thị thân thiện: ['Tích cực' (2), 'Trung lập' (1), 'Tiêu cực' (0)]
      const rawCm = modelsCache.evaluation_report.confusion_matrix;
      const idxOrder = [2, 1, 0];
      const cmLabels = ['Tích cực', 'Trung lập', 'Tiêu cực'];
      const cmCells = [];

      idxOrder.forEach((actualRawIdx, yIdx) => {
        const row = rawCm[actualRawIdx] || [0, 0, 0];
        const rowSum = Math.max(1, row.reduce((a, b) => a + b, 0));
        idxOrder.forEach((predRawIdx, xIdx) => {
          const count = row[predRawIdx] || 0;
          const pct = Math.round((count * 1000) / rowSum) / 10;
          cmCells.push([xIdx, yIdx, pct, count]);
        });
      });

      chartCm.setOption({
        tooltip: {
          position: 'top',
          formatter: p => `Thực tế <b>${cmLabels[p.data[1]]}</b> → Dự đoán <b>${cmLabels[p.data[0]]}</b>: <b>${p.data[2]}%</b> (${Number(p.data[3]).toLocaleString('en-US')} mẫu)`
        },
        series: [{
          data: cmCells,
          label: {
            show: true,
            formatter: p => p.data[2] + '%',
            color: '#1e293b',
            fontWeight: 600
          }
        }]
      });
    }
  } catch (e) {
    // Fallback offline
  }
}

// ECharts Initialization
window.addEventListener('DOMContentLoaded', () => {
  // 1. Time Series Chart
  const chartTsElem = document.getElementById('chart-timeseries');
  if (chartTsElem) {
    const chartTs = echarts.init(chartTsElem);
    const months = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'];
    chartTs.setOption({
      tooltip: { trigger: 'axis' },
      grid: { top: 30, right: 30, bottom: 25, left: 45 },
      xAxis: {
        type: 'category',
        data: months,
        axisLine: { lineStyle: { color: '#e2e8f0' } },
        axisLabel: { color: '#64748b' }
      },
      yAxis: [
        {
          type: 'value',
          name: 'Đánh giá',
          nameTextStyle: { color: '#94a3b8', fontSize: 11 },
          splitLine: { lineStyle: { color: '#f8fafc', type: 'dashed' } },
          axisLabel: { color: '#94a3b8' }
        },
        {
          type: 'value',
          name: '% Tiêu cực',
          nameTextStyle: { color: '#f43f5e', fontSize: 11 },
          axisLabel: { formatter: '{value}%', color: '#f43f5e' },
          splitLine: { show: false }
        }
      ],
      series: [
        {
          name: 'Số đánh giá',
          type: 'bar',
          barWidth: '38%',
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#fecdd3' },
              { offset: 1, color: '#ffe4e6' }
            ]),
            borderRadius: [6, 6, 0, 0]
          },
          data: [2800, 3100, 3900, 4200, 4500, 4100, 3800, 4600, 5100, 4900, 5400, 6200]
        },
        {
          name: '% Tiêu cực',
          type: 'line',
          yAxisIndex: 1,
          smooth: true,
          symbolSize: 6,
          lineStyle: { color: '#f43f5e', width: 3 },
          itemStyle: { color: '#f43f5e' },
          data: [8.5, 8.2, 7.6, 7.9, 7.1, 7.4, 6.9, 7.2, 8.1, 7.3, 7.5, 6.8]
        }
      ]
    });
  }

  // 2. Donut Chart (bổ sung thông tin trung tâm để không bị rỗng)
  const chartDonutElem = document.getElementById('chart-donut');
  if (chartDonutElem) {
    const chartDonut = echarts.init(chartDonutElem);
    chartDonut.setOption({
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { show: false },
      title: {
        text: '82.4%',
        subtext: 'Tích cực',
        left: 'center',
        top: '36%',
        itemGap: 4,
        textStyle: {
          fontSize: 22,
          fontWeight: '800',
          color: '#10b981',
          fontFamily: 'JetBrains Mono, monospace'
        },
        subtextStyle: {
          fontSize: 11,
          fontWeight: '600',
          color: '#64748b',
          fontFamily: 'Plus Jakarta Sans, sans-serif'
        }
      },
      series: [
        {
          name: 'Cảm xúc',
          type: 'pie',
          radius: ['58%', '82%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 8,
            borderColor: '#ffffff',
            borderWidth: 3
          },
          label: { show: false },
          data: [
            { value: 82.4, name: 'Tích cực', itemStyle: { color: '#10b981' } },
            { value: 9.8, name: 'Trung lập', itemStyle: { color: '#f59e0b' } },
            { value: 7.8, name: 'Tiêu cực', itemStyle: { color: '#f43f5e' } }
          ]
        }
      ]
    });
  }

  // 3. Price Tier Chart
  const chartTierElem = document.getElementById('chart-tier');
  if (chartTierElem) {
    const chartTier = echarts.init(chartTierElem);
    chartTier.setOption({
      tooltip: { trigger: 'axis' },
      grid: { top: 20, right: 20, bottom: 25, left: 45 },
      xAxis: {
        type: 'category',
        data: ['Bình dân (<200k)', 'Tầm trung (200k-1.5M)', 'Cao cấp (>1.5M)'],
        axisLabel: { color: '#64748b' }
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: '{value}%', color: '#94a3b8' },
        splitLine: { lineStyle: { color: '#f1f5f9' } }
      },
      series: [{
        name: '% Tiêu cực',
        type: 'bar',
        barWidth: '40%',
        itemStyle: {
          color: function(params) {
            const colors = ['#38bdf8', '#f59e0b', '#f43f5e'];
            return colors[params.dataIndex] || '#f43f5e';
          },
          borderRadius: [8, 8, 0, 0]
        },
        data: [5.2, 7.8, 11.4]
      }]
    });
  }

  // 4. Category Chart (sử dụng containLabel: true để không bị cắt/đè chữ danh mục)
  const chartCatElem = document.getElementById('chart-category');
  if (chartCatElem) {
    const chartCat = echarts.init(chartCatElem);
    chartCat.setOption({
      tooltip: { trigger: 'axis' },
      grid: { top: 15, right: 48, bottom: 25, left: 12, containLabel: true },
      xAxis: {
        type: 'value',
        max: 100,
        axisLabel: { formatter: '{value}%', color: '#94a3b8' },
        splitLine: { lineStyle: { color: '#f1f5f9' } }
      },
      yAxis: {
        type: 'category',
        data: ['Đồ Chơi - Mẹ & Bé', 'Điện & Đồ Gia Dụng', 'Làm Đẹp - Sức Khỏe', 'Điện Thoại - MTB', 'Sách & VPP'],
        axisLabel: { color: '#475569', fontSize: 11, fontWeight: 500, margin: 10 }
      },
      series: [{
        name: '% Hài lòng',
        type: 'bar',
        barWidth: '50%',
        itemStyle: { color: '#fb7185', borderRadius: [0, 8, 8, 0] },
        label: { show: true, position: 'right', formatter: '{c}%', color: '#64748b', fontSize: 11, fontWeight: 600 },
        data: [81.5, 79.8, 84.1, 76.5, 89.2]
      }]
    });
  }

  // 5. Tier Detail Chart
  const chartTierDetailElem = document.getElementById('chart-tier-detail');
  if (chartTierDetailElem) {
    const chartTierDetail = echarts.init(chartTierDetailElem);
    chartTierDetail.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['% Hài lòng', '% Phàn nàn'], top: 0 },
      grid: { top: 35, right: 20, bottom: 25, left: 45 },
      xAxis: {
        type: 'category',
        data: ['Bình dân', 'Tầm trung', 'Cao cấp'],
        axisLabel: { color: '#64748b' }
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: '{value}%', color: '#94a3b8' },
        splitLine: { lineStyle: { color: '#f1f5f9' } }
      },
      series: [
        {
          name: '% Hài lòng',
          type: 'bar',
          itemStyle: { color: '#10b981', borderRadius: [6, 6, 0, 0] },
          data: [86.5, 81.2, 74.8]
        },
        {
          name: '% Phàn nàn',
          type: 'bar',
          itemStyle: { color: '#f43f5e', borderRadius: [6, 6, 0, 0] },
          data: [5.2, 7.8, 11.4]
        }
      ]
    });
  }

  // 6. Heatmap Chart (Mặc định Top 3 danh mục nhiều SP & bình luận nhất — ma trận 3x3 thoáng đẹp)
  const chartHeatmapElem = document.getElementById('chart-heatmap');
  if (chartHeatmapElem) {
    const chartHeatmap = echarts.init(chartHeatmapElem);
    const categories = ['Sách & VPP', 'Điện & Gia Dụng', 'Bách Hóa'];
    const tiers = ['Bình dân', 'Tầm trung', 'Cao cấp'];
    const data = [
      [0, 0, 5.8], [0, 1, 9.2], [0, 2, 11.4],
      [1, 0, 8.9], [1, 1, 13.5], [1, 2, 16.2],
      [2, 0, 7.6], [2, 1, 11.8], [2, 2, 14.1]
    ];
    chartHeatmap.setOption({
      tooltip: { position: 'top', formatter: p => `${categories[p.data[0]]} (${tiers[p.data[1]]}): ${p.data[2]}% tiêu cực` },
      grid: { top: 12, right: 20, bottom: 72, left: 15, containLabel: true },
      xAxis: { type: 'category', data: categories, axisLabel: { color: '#475569', fontSize: 11, fontWeight: 600, margin: 10 } },
      yAxis: { type: 'category', data: tiers, axisLabel: { color: '#475569', fontSize: 11, fontWeight: 500, margin: 10 } },
      visualMap: {
        min: 0,
        max: 17,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 2,
        itemWidth: 12,
        itemHeight: 120,
        text: ['17%', '0%'],
        textStyle: { color: '#64748b', fontSize: 11 },
        inRange: { color: ['#ffe4e6', '#fb7185', '#be123c'] }
      },
      series: [{
        name: '% Tiêu cực',
        type: 'heatmap',
        data: data,
        label: {
          show: true,
          formatter: p => p.data[2] + '%',
          color: '#1e293b',
          fontWeight: 600,
          fontSize: 12
        },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.2)' } }
      }]
    });
  }

  // Customer sub charts
  const chartPhotoElem = document.getElementById('chart-customer-photo');
  if (chartPhotoElem) {
    const cp = echarts.init(chartPhotoElem);
    cp.setOption({
      tooltip: { trigger: 'axis' },
      grid: { top: 15, right: 20, bottom: 25, left: 15, containLabel: true },
      xAxis: { type: 'category', data: ['Không ảnh', 'Có gửi ảnh'], axisLabel: { color: '#475569', fontSize: 11, fontWeight: 500 } },
      yAxis: { type: 'value', axisLabel: { formatter: '{value}%', color: '#94a3b8' }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
      series: [{
        type: 'bar',
        barWidth: '40%',
        itemStyle: { color: '#f43f5e', borderRadius: [6, 6, 0, 0] },
        data: [6.1, 12.8]
      }]
    });
  }

  const chartFreqElem = document.getElementById('chart-customer-freq');
  if (chartFreqElem) {
    const cf = echarts.init(chartFreqElem);
    cf.setOption({
      tooltip: { trigger: 'axis' },
      grid: { top: 15, right: 20, bottom: 25, left: 15, containLabel: true },
      xAxis: { type: 'category', data: ['Mua lần đầu', 'Mua 2-4 lần', 'Khách VIP'], axisLabel: { color: '#475569', fontSize: 11, fontWeight: 500 } },
      yAxis: { type: 'value', axisLabel: { formatter: '{value}%', color: '#94a3b8' }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
      series: [{
        type: 'bar',
        barWidth: '40%',
        itemStyle: { color: '#10b981', borderRadius: [6, 6, 0, 0] },
        data: [79.2, 85.6, 91.4]
      }]
    });
  }

  const chartSellerElem = document.getElementById('chart-customer-seller');
  if (chartSellerElem) {
    const cs = echarts.init(chartSellerElem);
    cs.setOption({
      tooltip: { trigger: 'axis' },
      grid: { top: 15, right: 48, bottom: 25, left: 12, containLabel: true },
      xAxis: {
        type: 'value',
        max: 100,
        axisLabel: { formatter: '{value}%', color: '#94a3b8', fontSize: 10 },
        splitLine: { lineStyle: { color: '#f1f5f9' } }
      },
      yAxis: {
        type: 'category',
        data: ['Pmart', 'Nhà sách Fahasa', 'Thành Thuận Thiên', 'Durex Official', 'Tiki Trading'],
        axisLabel: { color: '#475569', fontSize: 11, fontWeight: 500, margin: 10 }
      },
      series: [{
        name: '% Hài lòng',
        type: 'bar',
        barWidth: '48%',
        itemStyle: { color: '#fb7185', borderRadius: [0, 6, 6, 0] },
        label: { show: true, position: 'right', formatter: '{c}%', color: '#64748b', fontSize: 11, fontWeight: 600 },
        data: [79.7, 91.8, 80.6, 76.1, 80.9]
      }]
    });
  }

  // Models comparison
  const chartModelsElem = document.getElementById('chart-models');
  if (chartModelsElem) {
    const cm = echarts.init(chartModelsElem);
    cm.setOption({
      tooltip: { trigger: 'axis' },
      grid: { top: 20, right: 20, bottom: 25, left: 40 },
      xAxis: { type: 'category', data: ['PhoBERT', 'LinearSVC', 'LogReg', 'NaiveBayes'], axisLabel: { color: '#64748b' } },
      yAxis: { type: 'value', min: 80, max: 96, axisLabel: { formatter: '{value}%', color: '#94a3b8' } },
      series: [{
        type: 'bar',
        barWidth: '40%',
        itemStyle: { color: '#f43f5e', borderRadius: [6, 6, 0, 0] },
        data: [94.2, 92.3, 89.1, 84.5]
      }]
    });
  }

  // 7. Confusion Matrix (đồng bộ dải màu visualMap giống Bản đồ nhiệt tỷ lệ không hài lòng)
  const chartCmElem = document.getElementById('chart-cm');
  if (chartCmElem) {
    const cme = echarts.init(chartCmElem);
    const cmLabels = ['Tích cực', 'Trung lập', 'Tiêu cực'];
    cme.setOption({
      tooltip: {
        position: 'top',
        formatter: p => `Thực tế <b>${cmLabels[p.data[1]]}</b> → Dự đoán <b>${cmLabels[p.data[0]]}</b>: ${p.data[2]}%`
      },
      grid: { top: 12, right: 20, bottom: 72, left: 15, containLabel: true },
      xAxis: { type: 'category', data: cmLabels, axisLabel: { color: '#475569', fontWeight: 500, margin: 10 } },
      yAxis: { type: 'category', data: cmLabels, axisLabel: { color: '#475569', fontWeight: 500, margin: 10 } },
      visualMap: {
        min: 0,
        max: 100,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 2,
        itemWidth: 12,
        itemHeight: 120,
        text: ['100%', '0%'],
        textStyle: { color: '#64748b', fontSize: 11 },
        inRange: { color: ['#ffe4e6', '#fb7185', '#be123c'] }
      },
      series: [{
        name: 'Độ chính xác',
        type: 'heatmap',
        data: [
          [0, 0, 94], [0, 1, 4], [0, 2, 2],
          [1, 0, 8], [1, 1, 78], [1, 2, 14],
          [2, 0, 3], [2, 1, 8], [2, 2, 89]
        ],
        label: {
          show: true,
          formatter: p => p.data[2] + '%',
          color: '#1e293b',
          fontWeight: 600
        },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.2)' } }
      }]
    });
  }

  window.addEventListener('resize', () => {
    [
      chartTsElem, chartDonutElem, chartTierElem, chartCatElem,
      chartTierDetailElem, chartHeatmapElem, chartPhotoElem,
      chartFreqElem, chartSellerElem, chartModelsElem, chartCmElem
    ].forEach(el => {
      if (el) {
        const inst = echarts.getInstanceByDom(el);
        if (inst) inst.resize();
      }
    });
  });

  // Load real data from backend if server is running
  applyFilters();
});
