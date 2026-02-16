// VeloPark – Enhanced Frontend JavaScript v4
// =============================================

let currentUser = null;
let token = null;
let userChart = null;
let adminChart = null;
let revenueChart = null;
let mostBookedChart = null;
let dlForecastChart = null; // Deprecated, kept for safety or overwrite
let autoRefreshTimer = null;
let liveCostTimers = {}; // Store timers for live cost updates
let lotsData = [];

// ── Toast Notification ──────────────────────────────────
function showToast(message, type = 'primary') {
  const container = document.getElementById('toast-container');
  const icons = {
    success: 'bi-check-circle-fill',
    danger: 'bi-exclamation-triangle-fill',
    primary: 'bi-info-circle-fill',
    info: 'bi-info-circle-fill',
    warning: 'bi-exclamation-circle-fill'
  };
  const toast = document.createElement('div');
  toast.className = `toast align-items-center text-bg-${type} border-0`;
  toast.setAttribute('role', 'alert');
  toast.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">
        <i class="bi ${icons[type] || 'bi-info-circle-fill'} me-2"></i>${message}
      </div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
    </div>
  `;
  container.appendChild(toast);
  const bsToast = new bootstrap.Toast(toast, { delay: 3500 });
  bsToast.show();
  toast.addEventListener('hidden.bs.toast', () => toast.remove());
}

// ── API Call Helper ─────────────────────────────────────
async function api(path, options = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { ...headers, ...(options.headers || {}) }
  });

  if (!response.ok) {
    let errorMsg = response.statusText;
    try {
      const errorData = await response.json();
      errorMsg = errorData.message || errorData.error || errorMsg;
    } catch { }
    throw new Error(errorMsg);
  }

  return response.json();
}

// ── Animated Counter ────────────────────────────────────
function animateValue(elementId, endValue, prefix = '', suffix = '', duration = 600) {
  const el = document.getElementById(elementId);
  if (!el) return;

  if (isNaN(endValue)) {
    el.textContent = endValue || '—';
    return;
  }

  const startValue = parseInt(el.textContent.replace(/[^0-9.-]/g, '')) || 0;
  const diff = endValue - startValue;
  if (diff === 0) {
    el.textContent = `${prefix}${endValue}${suffix}`;
    return;
  }

  const startTime = performance.now();
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(startValue + diff * eased);
    el.textContent = `${prefix}${current}${suffix}`;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ── Section Visibility ──────────────────────────────────
function showSection(section) {
  const sections = ['auth-section', 'user-dashboard', 'admin-dashboard'];
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.style.display = id === section ? 'block' : 'none';
      if (id === section) {
        el.style.opacity = '0';
        el.style.transform = 'translateY(12px)';
        requestAnimationFrame(() => {
          el.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
          el.style.opacity = '1';
          el.style.transform = 'translateY(0)';
        });
      }
    }
  });
}

function updateUserInfo() {
  const userInfo = document.getElementById('user-info');
  if (currentUser) {
    const roleColor = currentUser.role === 'admin' ? 'var(--accent-rose)' : 'var(--accent-indigo)';
    userInfo.innerHTML = `
      <span class="me-2" style="color: var(--text-secondary); font-size: 0.9rem;">
        <i class="bi bi-person-circle me-1" style="color: ${roleColor};"></i>
        <strong style="color: var(--text-primary);">${currentUser.username}</strong>
        <span class="badge ms-1">${currentUser.role}</span>
      </span>
      <button class="btn btn-outline-danger btn-sm" onclick="logout()" id="logout-btn">
        <i class="bi bi-box-arrow-right me-1"></i>Logout
      </button>
    `;

    if (currentUser.role === 'admin') {
      showSection('admin-dashboard');
      loadAdminLots();
      loadAdminMetrics();
      loadAdminRevenue();
      loadReports();
      loadMostBookedLots();
      loadMLStatus();
      startAutoRefresh('admin');
    } else {
      showSection('user-dashboard');
      document.getElementById('user-greeting').textContent = `Welcome back, ${currentUser.username}`;
      loadUserData();
      startAutoRefresh('user');
    }
  } else {
    userInfo.innerHTML = '';
    showSection('auth-section');
    stopAutoRefresh();
  }
}

// ── Auto Refresh ────────────────────────────────────────
function startAutoRefresh(mode) {
  stopAutoRefresh();
  autoRefreshTimer = setInterval(() => {
    if (mode === 'admin') {
      loadAdminLots();
      loadAdminMetrics();
      loadAdminRevenue();
    } else {
      loadUserData();
    }
  }, 30000);
}

function stopAutoRefresh() {
  if (autoRefreshTimer) {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
}

// ── Auth Functions ──────────────────────────────────────
async function login(e) {
  e.preventDefault();
  const btn = document.getElementById('login-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const data = await api('/auth/login', {
      method: 'POST',
      body: JSON.stringify({
        username: document.getElementById('login-username').value,
        password: document.getElementById('login-password').value
      })
    });
    token = data.access_token;
    currentUser = data.user;
    localStorage.setItem('token', token);
    updateUserInfo();
    showToast(`Welcome back, ${currentUser.username}!`, 'success');
  } catch (error) {
    showToast(error.message, 'danger');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

async function register(e) {
  e.preventDefault();
  const btn = document.getElementById('register-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    await api('/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        username: document.getElementById('register-username').value,
        email: document.getElementById('register-email').value,
        password: document.getElementById('register-password').value
      })
    });
    showToast('Account created! Please sign in.', 'success');
    document.getElementById('register-form').reset();
  } catch (error) {
    showToast(error.message, 'danger');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

function logout() {
  api('/auth/logout', { method: 'POST' }).catch(() => { });
  token = null;
  currentUser = null;
  localStorage.removeItem('token');
  if (userChart) { userChart.destroy(); userChart = null; }
  if (adminChart) { adminChart.destroy(); adminChart = null; }
  if (revenueChart) { revenueChart.destroy(); revenueChart = null; }
  if (mostBookedChart) { mostBookedChart.destroy(); mostBookedChart = null; }
  updateUserInfo();
  showToast('Signed out successfully', 'info');
}

// ── User Dashboard ──────────────────────────────────────
async function loadUserData() {
  try {
    const [lots, reservations, metrics] = await Promise.all([
      api('/lots'),
      api('/reservations/me'),
      api('/metrics/user')
    ]);

    lotsData = lots;

    // Populate lot select
    const select = document.getElementById('lot-select');
    const currentVal = select.value;
    select.innerHTML = '<option value="">— Choose a lot —</option>';
    lots.forEach(lot => {
      const option = document.createElement('option');
      option.value = lot.id;
      const pct = lot.total > 0 ? Math.round((lot.available / lot.total) * 100) : 0;
      option.textContent = `${lot.prime_location_name} — ${lot.available}/${lot.total} available (${pct}%)`;
      if (lot.available === 0) option.disabled = true;
      select.appendChild(option);
    });
    if (currentVal) select.value = currentVal;

    // Fetch AI recommendations and badge the best lot
    fetchRecommendations(lots, select);

    // Clear existing live cost timers
    Object.values(liveCostTimers).forEach(clearTimeout);
    liveCostTimers = {};

    // Reservations table
    const tbody = document.getElementById('reservations-table');
    if (reservations.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7">
        <div class="empty-state">
          <i class="bi bi-inbox"></i>
          <p>No reservations yet.<br/>Book a spot to get started!</p>
        </div>
      </td></tr>`;
    } else {
      tbody.innerHTML = reservations.map(r => {
        const isActive = !r.end_time;
        const lotInfo = lots.find(l => l.id === r.lot_id);
        const lotName = lotInfo ? lotInfo.prime_location_name : `Lot #${r.lot_id}`;
        const startDate = new Date(r.start_time);
        const timeAgo = getTimeAgo(startDate);
        return `
        <tr>
          <td><span class="badge bg-primary">#${r.id}</span></td>
          <td><strong>${lotName}</strong></td>
          <td><span class="badge bg-success">S${r.index_number}</span></td>
          <td>
            <small style="color: var(--text-secondary);">${startDate.toLocaleDateString()}</small><br/>
            <small style="color: var(--text-muted);">${timeAgo}</small>
          </td>
          <td><strong style="color: ${r.parking_cost ? 'var(--accent-amber)' : 'var(--text-muted)'};">
            ${r.parking_cost != null
            ? '₹' + r.parking_cost.toFixed(2)
            : isActive
              ? `<span id="live-cost-${r.id}" class="live-cost-badge">Calculate...</span>`
              : '—'}
          </strong></td>
          <td>
            ${isActive
            ? '<span class="badge" style="background: rgba(16,185,129,0.15); color: var(--accent-emerald); border: 1px solid rgba(16,185,129,0.3);"><i class="bi bi-circle-fill me-1" style="font-size: 0.5rem;"></i>Active</span>'
            : '<span class="badge" style="background: rgba(100,116,139,0.15); color: var(--text-muted); border: 1px solid rgba(100,116,139,0.2);">Completed</span>'
          }
          </td>
          <td>
            ${isActive ? `<button class="btn btn-success btn-sm" onclick="releaseSpot(${r.id})" id="release-btn-${r.id}">
              <i class="bi bi-unlock me-1"></i>Release
            </button>` : ''}
          </td>
        </tr>`;
      }).join('');

      // Start live cost updates for active reservations
      reservations.forEach(r => {
        if (!r.end_time && r.price_per_hour) {
          startLiveCostUpdates(r.id, r.start_time, r.price_per_hour);
        }
      });
    }

    // Animate metrics
    animateValue('metric-total', metrics.total_reservations);
    animateValue('metric-active', metrics.active_reservations);
    animateValue('metric-spent', Math.round(metrics.amount_spent_30_days), '₹');
    document.getElementById('metric-fav').textContent = metrics.favorite_lot || '—';

    // User availability chart
    renderUserChart(lots);

  } catch (error) {
    showToast('Error loading data: ' + error.message, 'danger');
  }
}

// ── Live Cost Update Mechanism ──────────────────────────
function startLiveCostUpdates(resId, startTimeStr, pricePerHour) {
  const el = document.getElementById(`live-cost-${resId}`);
  if (!el) return;

  // Use local time parsing to match backend's datetime.now()
  const startTime = new Date(startTimeStr);
  const ratePerMs = pricePerHour / 3600000;

  function update() {
    const now = new Date();
    const elapsedMs = Math.max(0, now - startTime);
    const cost = elapsedMs * ratePerMs;

    // Display current cost
    // "Updated every time 1 rupee is incremented" - we show integer or 2 decimals?
    // User asked: "updated every time 1 rupee is incremented"
    // We'll show rounded integer or high precision if they want 'odometer',
    // but sticking to standard currency format (₹120) usually implies integer for 'odometer'.
    // Let's show rounded down integer to match the "1 rupee increment" feel, or 2 decimals if required.
    // Given the prompt, I'll show integer for the "rupee" effect, or maybe just 2 decimals but update rarely?
    // Strategy: Update precisely when cost crosses integer boundary.
    el.innerHTML = `<span class="odometer-anim">₹${Math.floor(cost)}</span>`;
    el.style.color = 'var(--accent-amber)';
    el.style.fontWeight = '700';

    // Calculate time until next rupee
    const nextRupeeCost = Math.floor(cost) + 1;
    const nextRupeeMs = nextRupeeCost / ratePerMs;
    const delay = Math.max(50, nextRupeeMs - elapsedMs); // Min 50ms delay to prevent freeze

    liveCostTimers[resId] = setTimeout(update, delay);
  }

  update();
}

function getTimeAgo(date) {
  const seconds = Math.floor((new Date() - date) / 1000);
  if (seconds < 60) return 'Just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

// ── Lot Preview ─────────────────────────────────────────
function onLotSelect() {
  const select = document.getElementById('lot-select');
  const preview = document.getElementById('lot-preview');
  const val = parseInt(select.value);

  if (!val || !lotsData.length) {
    preview.style.display = 'none';
    return;
  }

  const lot = lotsData.find(l => l.id === val);
  if (!lot) { preview.style.display = 'none'; return; }

  preview.style.display = 'block';
  const pct = lot.total > 0 ? Math.round(((lot.total - lot.available) / lot.total) * 100) : 0;
  document.getElementById('lot-preview-avail').textContent = `${lot.available} / ${lot.total} spots`;
  document.getElementById('lot-preview-price').textContent = lot.price_per_hour;

  const bar = document.getElementById('lot-preview-bar');
  bar.style.width = `${pct}%`;
  bar.className = 'occupancy-bar-fill';
  if (pct > 80) bar.classList.add('high');
  else if (pct > 50) bar.classList.add('medium');
  else bar.classList.add('low');

  document.getElementById('lot-preview-est').textContent = `₹${lot.price_per_hour}/hr`;

  // Fetch AI duration prediction
  fetchAIPrediction(val);
}

// ── User Chart ──────────────────────────────────────────
function renderUserChart(lots) {
  const ctx = document.getElementById('userAvailChart');
  if (!ctx) return;
  if (userChart) userChart.destroy();

  userChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: lots.map(l => l.prime_location_name),
      datasets: [
        {
          label: 'Available',
          data: lots.map(l => l.available),
          backgroundColor: 'rgba(16, 185, 129, 0.6)',
          borderColor: 'rgba(16, 185, 129, 1)',
          borderWidth: 1,
          borderRadius: 6,
        },
        {
          label: 'Occupied',
          data: lots.map(l => l.total - l.available),
          backgroundColor: 'rgba(244, 63, 94, 0.5)',
          borderColor: 'rgba(244, 63, 94, 1)',
          borderWidth: 1,
          borderRadius: 6,
        }
      ]
    },
    options: chartOptions()
  });
}

// ── Booking ─────────────────────────────────────────────
async function bookSpot(e) {
  e.preventDefault();
  const btn = document.getElementById('book-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const res = await api('/reservations', {
      method: 'POST',
      body: JSON.stringify({
        lot_id: document.getElementById('lot-select').value,
        vehicle_number: document.getElementById('vehicle-number').value,
        vehicle_model: document.getElementById('vehicle-model').value
      })
    });
    showToast(`Spot S${res.index_number} reserved successfully!`, 'success');
    document.getElementById('book-form').reset();
    document.getElementById('lot-preview').style.display = 'none';
    loadUserData();
  } catch (error) {
    showToast(error.message, 'danger');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

async function releaseSpot(id) {
  const btn = document.getElementById(`release-btn-${id}`);
  if (btn) { btn.classList.add('loading'); btn.disabled = true; }

  try {
    const res = await api(`/reservations/${id}/release`, { method: 'POST' });
    showToast(`Spot released! Cost: ₹${res.parking_cost.toFixed(2)}`, 'success');
    loadUserData();
  } catch (error) {
    showToast(error.message, 'danger');
  } finally {
    if (btn) { btn.classList.remove('loading'); btn.disabled = false; }
  }
}

// ── CSV Export (Direct Download) ────────────────────────
async function triggerExport() {
  const btn = document.getElementById('export-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const savedToken = localStorage.getItem('token');
    const response = await fetch('/api/export/csv/download', {
      headers: { 'Authorization': `Bearer ${savedToken}` }
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.message || 'Export failed');
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'reservations-export.csv';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    showToast('CSV exported successfully!', 'success');
  } catch (error) {
    showToast('Export failed: ' + error.message, 'danger');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ── Admin Functions ─────────────────────────────────────
async function loadAdminLots() {
  try {
    const lots = await api('/admin/lots');
    const tbody = document.getElementById('admin-lots-table');

    if (lots.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6">
        <div class="empty-state" style="padding: 1.5rem;">
          <i class="bi bi-building" style="font-size: 2rem;"></i>
          <p>No lots created yet</p>
        </div>
      </td></tr>`;
    } else {
      tbody.innerHTML = lots.map(lot => {
        const total = lot.available + lot.occupied;
        const pct = total > 0 ? Math.round((lot.occupied / total) * 100) : 0;
        const barClass = pct > 80 ? 'high' : pct > 50 ? 'medium' : 'low';
        return `
        <tr>
          <td><strong>${lot.prime_location_name}</strong></td>
          <td style="color: var(--accent-amber);"><strong>₹${lot.price_per_hour}</strong>/hr</td>
          <td style="min-width: 120px;">
            <div class="occupancy-bar-wrap">
              <div class="occupancy-bar-fill ${barClass}" style="width: ${pct}%;"></div>
            </div>
            <small style="color: var(--text-muted);">${pct}% full</small>
          </td>
          <td><span class="badge bg-success">${lot.available}</span></td>
          <td><span class="badge bg-danger">${lot.occupied}</span></td>
          <td>
            <div class="d-flex gap-1">
              <button class="btn btn-outline-primary btn-sm" onclick="viewLotDetails(${lot.id})" title="View Details">
                <i class="bi bi-eye"></i>
              </button>
              <button class="btn btn-outline-danger btn-sm" onclick="deleteLot(${lot.id})" title="Delete Lot">
                <i class="bi bi-trash"></i>
              </button>
            </div>
          </td>
        </tr>`;
      }).join('');
    }

    renderAdminChart(lots);
  } catch (error) {
    showToast('Error loading lots: ' + error.message, 'danger');
  }
}

async function loadAdminMetrics() {
  try {
    const data = await api('/metrics/admin');
    animateValue('admin-metric-users', data.totals.users);
    animateValue('admin-metric-lots', data.totals.lots);
    animateValue('admin-metric-spots', data.totals.spots);
    animateValue('admin-metric-occupied', data.totals.occupied);
    if (data.totals.total_revenue !== undefined) {
      animateValue('admin-metric-revenue', Math.round(data.totals.total_revenue), '₹');
    }
  } catch (error) {
    console.warn('Failed to load admin metrics:', error);
  }
}

// ── Admin Revenue (Pie Chart) ───────────────────────────
async function loadAdminRevenue() {
  try {
    const data = await api('/metrics/admin/revenue');

    const badge = document.getElementById('revenue-total-badge');
    badge.textContent = `₹${data.total_revenue.toFixed(2)} total`;

    const emptyEl = document.getElementById('revenue-chart-empty');
    const wrapEl = document.getElementById('revenue-chart-wrap');
    const breakdownEl = document.getElementById('revenue-breakdown');

    if (data.per_user.length === 0) {
      emptyEl.style.display = 'block';
      wrapEl.style.display = 'none';
      breakdownEl.innerHTML = '';
      return;
    }

    emptyEl.style.display = 'none';
    wrapEl.style.display = 'block';

    renderRevenueChart(data.per_user, data.total_revenue);

    breakdownEl.innerHTML = `
      <div class="table-responsive" style="border-radius: 8px; border: 1px solid var(--border-color); overflow: hidden;">
        <table class="table table-sm mb-0">
          <thead>
            <tr>
              <th></th>
              <th>User</th>
              <th>Amount</th>
              <th>Share</th>
            </tr>
          </thead>
          <tbody>
            ${data.per_user.map((u, i) => {
      const share = data.total_revenue > 0 ? ((u.total_spent / data.total_revenue) * 100).toFixed(1) : 0;
      const color = PIE_COLORS[i % PIE_COLORS.length];
      return `
              <tr>
                <td><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${color};"></span></td>
                <td><strong>${u.username}</strong></td>
                <td style="color: var(--accent-amber);">₹${u.total_spent.toFixed(2)}</td>
                <td><span class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-secondary);">${share}%</span></td>
              </tr>`;
    }).join('')}
            <tr style="border-top: 1px solid var(--border-color);">
              <td></td>
              <td><strong style="color: var(--accent-cyan);">Total</strong></td>
              <td><strong style="color: var(--accent-cyan);">₹${data.total_revenue.toFixed(2)}</strong></td>
              <td><span class="badge" style="background: rgba(34,211,238,0.15); color: var(--accent-cyan);">100%</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    `;

  } catch (error) {
    console.warn('Failed to load admin revenue:', error);
  }
}

// Pie chart color palette
const PIE_COLORS = [
  '#818cf8', '#34d399', '#fbbf24', '#fb7185', '#22d3ee',
  '#a78bfa', '#f97316', '#2dd4bf', '#f472b6', '#60a5fa',
  '#4ade80', '#e879f9', '#facc15', '#38bdf8', '#a3e635'
];

// Polar / bar chart color palette
const POLAR_COLORS = [
  'rgba(99, 102, 241, 0.7)',
  'rgba(16, 185, 129, 0.7)',
  'rgba(245, 158, 11, 0.7)',
  'rgba(239, 68, 68, 0.7)',
  'rgba(6, 182, 212, 0.7)',
  'rgba(139, 92, 246, 0.7)',
  'rgba(236, 72, 153, 0.7)',
  'rgba(20, 184, 166, 0.7)',
];

const POLAR_BORDERS = [
  '#6366f1', '#10b981', '#f59e0b', '#ef4444',
  '#06b6d4', '#8b5cf6', '#ec4899', '#14b8a6',
];

function renderRevenueChart(perUser, totalRevenue) {
  const ctx = document.getElementById('revenueChart');
  if (!ctx) return;
  if (revenueChart) revenueChart.destroy();

  const labels = perUser.map(u => u.username);
  const values = perUser.map(u => u.total_spent);
  const colors = perUser.map((_, i) => PIE_COLORS[i % PIE_COLORS.length]);

  revenueChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: '#151c2c',
        borderWidth: 3,
        hoverBorderColor: '#1e293b',
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#94a3b8',
            font: { family: 'Inter', size: 12, weight: '500' },
            usePointStyle: true,
            pointStyle: 'circle',
            padding: 16,
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
          titleFont: { family: 'Inter', weight: '600' },
          bodyFont: { family: 'Inter' },
          callbacks: {
            label: function (ctx) {
              const value = ctx.parsed;
              const pct = totalRevenue > 0 ? ((value / totalRevenue) * 100).toFixed(1) : 0;
              return `  ₹${value.toFixed(2)} (${pct}%)`;
            }
          }
        }
      },
      animation: {
        animateRotate: true,
        animateScale: true,
        duration: 800,
        easing: 'easeOutQuart'
      }
    }
  });
}

// ── Admin Chart (Occupancy Bar) ─────────────────────────
function renderAdminChart(lots) {
  const ctx = document.getElementById('adminChart');
  if (!ctx) return;
  if (adminChart) adminChart.destroy();

  adminChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: lots.map(l => l.prime_location_name),
      datasets: [
        {
          label: 'Available',
          data: lots.map(l => l.available),
          backgroundColor: 'rgba(16, 185, 129, 0.6)',
          borderColor: 'rgba(16, 185, 129, 1)',
          borderWidth: 1,
          borderRadius: 8,
          borderSkipped: false,
        },
        {
          label: 'Occupied',
          data: lots.map(l => l.occupied),
          backgroundColor: 'rgba(244, 63, 94, 0.5)',
          borderColor: 'rgba(244, 63, 94, 1)',
          borderWidth: 1,
          borderRadius: 8,
          borderSkipped: false,
        }
      ]
    },
    options: chartOptions()
  });
}

function chartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#94a3b8',
          font: { family: 'Inter', weight: '500', size: 12 },
          usePointStyle: true,
          pointStyle: 'circle',
          padding: 16,
        }
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        titleFont: { family: 'Inter', weight: '600' },
        bodyFont: { family: 'Inter' },
      }
    },
    scales: {
      x: {
        ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
        grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false }
      },
      y: {
        beginAtZero: true,
        ticks: { color: '#64748b', font: { family: 'Inter', size: 11 }, stepSize: 1 },
        grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false }
      }
    },
    animation: { duration: 800, easing: 'easeOutQuart' }
  };
}

// ── Most Booked Lots (Polar Area Chart) ─────────────────
async function loadMostBookedLots() {
  try {
    const data = await api('/metrics/admin/most-booked');
    const emptyEl = document.getElementById('booked-chart-empty');
    const wrapEl = document.getElementById('booked-chart-wrap');

    if (!data.lots || data.lots.length === 0) {
      emptyEl.style.display = 'block';
      wrapEl.style.display = 'none';
      return;
    }

    emptyEl.style.display = 'none';
    wrapEl.style.display = 'block';

    renderMostBookedChart(data.lots);
  } catch (error) {
    console.warn('Failed to load most booked lots:', error);
  }
}

function renderMostBookedChart(lots) {
  const ctx = document.getElementById('mostBookedChart');
  if (!ctx) return;
  if (mostBookedChart) mostBookedChart.destroy();

  const labels = lots.map(l => l.lot);
  const values = lots.map(l => l.bookings);
  const bgColors = lots.map((_, i) => POLAR_COLORS[i % POLAR_COLORS.length]);
  const borderColors = lots.map((_, i) => POLAR_BORDERS[i % POLAR_BORDERS.length]);

  mostBookedChart = new Chart(ctx, {
    type: 'polarArea',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#94a3b8',
            font: { family: 'Inter', size: 12, weight: '500' },
            usePointStyle: true,
            pointStyle: 'circle',
            padding: 16,
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
          titleFont: { family: 'Inter', weight: '600' },
          bodyFont: { family: 'Inter' },
          callbacks: {
            label: function (ctx) {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              const pct = total > 0 ? ((ctx.parsed.r / total) * 100).toFixed(1) : 0;
              return `  ${ctx.parsed.r} booking${ctx.parsed.r !== 1 ? 's' : ''} (${pct}%)`;
            }
          }
        }
      },
      scales: {
        r: {
          ticks: {
            color: '#64748b',
            font: { family: 'Inter', size: 10 },
            stepSize: 1,
            backdropColor: 'transparent',
          },
          grid: { color: 'rgba(255,255,255,0.06)' },
          angleLines: { color: 'rgba(255,255,255,0.06)' },
        }
      },
      animation: {
        animateRotate: true,
        animateScale: true,
        duration: 1000,
        easing: 'easeOutQuart'
      }
    }
  });
}

// ── Admin CRUD ──────────────────────────────────────────
async function createLot(e) {
  e.preventDefault();
  const btn = document.getElementById('create-lot-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const name = document.getElementById('lot-name').value;
    const spots = parseInt(document.getElementById('lot-spots').value);
    await api('/admin/lots', {
      method: 'POST',
      body: JSON.stringify({
        prime_location_name: name,
        price_per_hour: parseInt(document.getElementById('lot-price').value),
        number_of_spots: spots
      })
    });
    showToast(`"${name}" created with ${spots} spots!`, 'success');
    document.getElementById('lot-form').reset();
    loadAdminLots();
    loadAdminMetrics();
  } catch (error) {
    showToast(error.message, 'danger');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

async function deleteLot(id) {
  if (!confirm('Are you sure you want to delete this parking lot?')) return;
  try {
    await api(`/admin/lots/${id}`, { method: 'DELETE' });
    showToast('Parking lot deleted', 'success');
    loadAdminLots();
    loadAdminMetrics();
  } catch (error) {
    showToast(error.message, 'danger');
  }
}

async function viewLotDetails(lotId) {
  try {
    const lot = await api(`/admin/lots/${lotId}`);
    const spots = lot.spots || [];
    let spotHTML = '<div class="spot-grid">';
    spots.forEach(s => {
      const cls = s.status === 'A' ? 'available' : 'occupied';
      const icon = s.status === 'A' ? '' : '<i class="bi bi-car-front-fill" style="font-size: 0.6rem;"></i>';
      let title = `Spot ${s.index_number} - ${s.status === 'A' ? 'Available' : 'Occupied'}`;
      if (s.current_reservation) {
        title += ` | ${s.current_reservation.username || 'User'} since ${new Date(s.current_reservation.start_time).toLocaleString()}`;
      }
      spotHTML += `<div class="spot-cell ${cls}" title="${title}">${icon || s.index_number}</div>`;
    });
    spotHTML += '</div>';

    const occupied = spots.filter(s => s.status === 'O').length;
    const total = spots.length;
    const pct = total > 0 ? Math.round((occupied / total) * 100) : 0;

    const overlay = document.createElement('div');
    overlay.id = 'lot-detail-overlay';
    overlay.style.cssText = `
      position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 1050;
      background: rgba(0,0,0,0.7); backdrop-filter: blur(8px);
      display: flex; justify-content: center; align-items: center;
      animation: fadeIn 0.2s ease-out;
    `;
    overlay.innerHTML = `
      <div style="background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 16px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; padding: 2rem; animation: slideUp 0.3s ease-out;">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <h5 style="font-weight: 700; margin: 0;">
            <i class="bi bi-building me-2" style="color: var(--accent-indigo);"></i>
            ${lot.prime_location_name}
          </h5>
          <button class="btn btn-outline-secondary btn-sm" onclick="document.getElementById('lot-detail-overlay').remove()">
            <i class="bi bi-x-lg"></i>
          </button>
        </div>
        <div class="d-flex gap-3 mb-3 flex-wrap">
          <div class="p-2 px-3" style="background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid var(--border-color);">
            <small style="color: var(--text-muted);">Price</small><br/>
            <strong style="color: var(--accent-amber);">₹${lot.price_per_hour}/hr</strong>
          </div>
          <div class="p-2 px-3" style="background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid var(--border-color);">
            <small style="color: var(--text-muted);">Occupancy</small><br/>
            <strong>${pct}%</strong> <small style="color: var(--text-muted);">(${occupied}/${total})</small>
          </div>
        </div>
        <div class="occupancy-bar-wrap mb-3">
          <div class="occupancy-bar-fill ${pct > 80 ? 'high' : pct > 50 ? 'medium' : 'low'}" style="width: ${pct}%;"></div>
        </div>
        <h6 style="font-weight: 600; color: var(--text-secondary); margin-bottom: 0.5rem;">
          <i class="bi bi-grid-3x3-gap me-1"></i> Spot Map
        </h6>
        <div class="d-flex gap-3 mb-2">
          <small><span class="spot-cell available" style="width: 16px; height: 16px; display: inline-flex; font-size: 0;"></span> Available</small>
          <small><span class="spot-cell occupied" style="width: 16px; height: 16px; display: inline-flex; font-size: 0;"></span> Occupied</small>
        </div>
        ${spotHTML}
      </div>
    `;
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.remove();
    });
    document.body.appendChild(overlay);
  } catch (error) {
    showToast('Error loading lot details: ' + error.message, 'danger');
  }
}

// ── Admin Tools ─────────────────────────────────────────
async function generateAdminReport() {
  const btn = document.getElementById('gen-report-btn');
  const status = document.getElementById('report-gen-status');
  const statusText = document.getElementById('report-gen-status-text');

  btn.classList.add('loading');
  btn.disabled = true;
  status.style.display = 'block';
  statusText.textContent = 'Generating report…';

  try {
    const data = await api('/admin/reports/generate', { method: 'POST' });
    showToast('Admin report generated successfully!', 'success');
    statusText.textContent = `Report ready: ${data.filename}`;
    status.className = 'alert alert-success mb-0 mt-2';
    downloadReport(data.filename);
    loadReports();
  } catch (error) {
    showToast('Report generation failed: ' + error.message, 'danger');
    statusText.textContent = 'Generation failed – ' + error.message;
    status.className = 'alert alert-danger mb-0 mt-2';
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

async function loadReports() {
  try {
    const data = await api('/admin/reports');
    const container = document.getElementById('reports-list');

    if (data.files.length === 0) {
      container.innerHTML = `
        <div class="empty-state" style="padding: 1rem;">
          <i class="bi bi-folder2-open" style="font-size: 1.5rem;"></i>
          <p style="font-size: 0.85rem;">No reports yet</p>
        </div>`;
    } else {
      // Show only the 3 most recent reports
      const recentFiles = data.files.slice(0, 3);
      container.innerHTML = `
        <div class="list-group">
          ${recentFiles.map(file => `
            <a href="#" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center"
               onclick="downloadReport('${file}'); return false;">
              <span><i class="bi bi-file-earmark-pdf-fill me-2" style="color: var(--accent-rose);"></i><small>${file}</small></span>
              <i class="bi bi-download" style="color: var(--accent-indigo);"></i>
            </a>
          `).join('')}
        </div>`;
    }
  } catch (error) {
    showToast('Error loading reports: ' + error.message, 'danger');
  }
}

async function downloadReport(filename) {
  try {
    const savedToken = localStorage.getItem('token');
    const response = await fetch(`/api/admin/reports/download?filename=${encodeURIComponent(filename)}`, {
      headers: { 'Authorization': `Bearer ${savedToken}` }
    });
    if (!response.ok) throw new Error('Download failed');
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    showToast('Report downloaded!', 'success');
  } catch (error) {
    showToast('Download failed: ' + error.message, 'danger');
  }
}

// ── ML / AI Functions ───────────────────────────────────
let recommendedLotId = null;

async function trainMLModels() {
  const btn = document.getElementById('ml-train-btn');
  const status = document.getElementById('ml-train-status');
  const statusText = document.getElementById('ml-train-status-text');
  btn.classList.add('loading');
  btn.disabled = true;
  status.style.display = 'block';
  statusText.textContent = 'Training models… this may take 10-30 seconds.';

  try {
    const result = await api('/ml/train', { method: 'POST' });
    const dMsg = result.duration?.message || result.duration?.error || 'Unknown';
    const rMsg = result.recommender?.message || result.recommender?.error || 'Unknown';
    const pMsg = result.pricing?.message || result.pricing?.error || 'Unknown';
    statusText.textContent = `Duration: ${dMsg} | Recommender: ${rMsg} | Pricing: ${pMsg}`;
    status.className = 'alert alert-success mb-3';
    showToast('All models trained successfully!', 'success');
    loadMLStatus();
    loadMLPricing();
  } catch (error) {
    statusText.textContent = 'Training failed: ' + error.message;
    status.className = 'alert alert-danger mb-3';
    showToast('ML training failed: ' + error.message, 'danger');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

async function loadMLStatus() {
  try {
    const s = await api('/ml/status');
    // sklearn badge
    const badge = document.getElementById('ml-sklearn-badge');
    if (s.sklearn_available) {
      badge.textContent = 'scikit-learn ✓';
      badge.style.background = 'rgba(16,185,129,0.15)';
      badge.style.color = 'var(--accent-emerald)';
      badge.style.borderColor = 'rgba(16,185,129,0.25)';
    } else {
      badge.textContent = 'sklearn missing';
      badge.style.background = 'rgba(244,63,94,0.15)';
      badge.style.color = 'var(--accent-rose)';
    }
    // Duration model
    const dStatus = document.getElementById('ml-duration-status');
    if (s.duration.trained) {
      dStatus.textContent = 'Trained ✓';
      dStatus.style.background = 'rgba(16,185,129,0.15)';
      dStatus.style.color = 'var(--accent-emerald)';
    }
    document.getElementById('ml-duration-samples').textContent = s.duration.samples;
    document.getElementById('ml-duration-mae').textContent = s.duration.mae_minutes != null ? s.duration.mae_minutes + ' min' : '—';
    // Recommender model
    const rStatus = document.getElementById('ml-recommender-status');
    if (s.recommender.trained) {
      rStatus.textContent = 'Trained ✓';
      rStatus.style.background = 'rgba(16,185,129,0.15)';
      rStatus.style.color = 'var(--accent-emerald)';
    }
    document.getElementById('ml-recommender-samples').textContent = s.recommender.samples;
    document.getElementById('ml-recommender-accuracy').textContent = s.recommender.accuracy != null ? s.recommender.accuracy + '%' : '—';
    // Dynamic Pricing model
    const pStatus = document.getElementById('ml-pricing-status');
    if (s.pricing && s.pricing.trained) {
      pStatus.textContent = 'Trained ✓';
      pStatus.style.background = 'rgba(16,185,129,0.15)';
      pStatus.style.color = 'var(--accent-emerald)';
      document.getElementById('ml-pricing-trained-text').textContent = 'Active';
      document.getElementById('ml-pricing-trained-text').style.color = 'var(--accent-emerald)';
      loadMLPricing();
    }
    if (s.pricing) {
      document.getElementById('ml-pricing-samples').textContent = s.pricing.samples || 0;
    }
  } catch {
    // Silently fail — ML status is optional
  }
}

async function fetchAIPrediction(lotId) {
  const panel = document.getElementById('ai-prediction-panel');
  if (!panel) return;
  try {
    const result = await api(`/ml/predict-duration?lot_id=${lotId}`);
    if (result.available) {
      panel.style.display = 'block';
      const mins = result.predicted_duration_minutes;
      let durationText;
      if (mins >= 60) {
        const hrs = Math.floor(mins / 60);
        const rem = Math.round(mins % 60);
        durationText = rem > 0 ? `${hrs}h ${rem}m` : `${hrs}h`;
      } else {
        durationText = `${Math.round(mins)}m`;
      }
      document.getElementById('ai-pred-duration').textContent = durationText;
      document.getElementById('ai-pred-cost').textContent = `₹${result.estimated_cost}`;
      const badge = document.getElementById('ai-confidence-badge');
      badge.textContent = result.confidence;
      if (result.confidence === 'high') {
        badge.style.background = 'rgba(16,185,129,0.15)';
        badge.style.color = 'var(--accent-emerald)';
      } else if (result.confidence === 'medium') {
        badge.style.background = 'rgba(251,191,36,0.15)';
        badge.style.color = 'var(--accent-amber)';
      } else {
        badge.style.background = 'rgba(100,116,139,0.15)';
        badge.style.color = 'var(--text-muted)';
      }
    } else {
      panel.style.display = 'none';
    }
  } catch {
    panel.style.display = 'none';
  }
}

async function fetchRecommendations(lots, selectEl) {
  try {
    const res = await api('/ml/recommend?top_n=1');
    if (res.available && res.recommendations.length > 0) {
      const topLot = res.recommendations[0];
      recommendedLotId = topLot.lot_id;
      // Update the matching option text with a star badge
      for (const option of selectEl.options) {
        if (parseInt(option.value) === topLot.lot_id) {
          option.textContent = '⭐ ' + option.textContent + ' — AI Recommended';
          break;
        }
      }
    }
  } catch {
    // Silently ignore — recommendations are optional
  }
}

async function loadMLPricing() {
  try {
    const res = await api('/ml/pricing');
    const container = document.getElementById('ml-pricing-table');
    if (!res.available || !res.suggestions || !res.suggestions.length) {
      container.innerHTML = `
        <div style="text-align: center; padding: 1rem; color: var(--text-muted);">
          <i class="bi bi-currency-exchange" style="font-size: 1.2rem;"></i>
          <p style="font-size: 0.8rem; margin-top: 0.4rem;">Train models to see pricing suggestions</p>
        </div>`;
      return;
    }
    container.innerHTML = `
      <table class="table table-sm mb-0" style="font-size: 0.78rem;">
        <thead>
          <tr>
            <th>Lot</th>
            <th>Base Price</th>
            ${res.suggestions[0].hourly_suggestions.map(h => `<th>${h.hour}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
          ${res.suggestions.map(s => `
            <tr>
              <td><strong>${s.lot_name}</strong></td>
              <td>₹${s.base_price}</td>
              ${s.hourly_suggestions.map(h => {
      const color = h.change_pct > 0 ? 'var(--accent-rose)' : h.change_pct < 0 ? 'var(--accent-emerald)' : 'var(--text-muted)';
      const arrow = h.change_pct > 0 ? '↑' : h.change_pct < 0 ? '↓' : '→';
      return `<td style="color: ${color};">₹${h.suggested_price}<br><small>${arrow}${Math.abs(h.change_pct)}%</small></td>`;
    }).join('')}
            </tr>
          `).join('')}
        </tbody>
      </table>`;
  } catch {
    // Silently fail
  }
}

// ── Initialize ──────────────────────────────────────────
async function init() {
  document.getElementById('login-form').addEventListener('submit', login);
  document.getElementById('register-form').addEventListener('submit', register);
  document.getElementById('book-form').addEventListener('submit', bookSpot);
  document.getElementById('lot-form').addEventListener('submit', createLot);
  document.getElementById('lot-select').addEventListener('change', onLotSelect);

  const savedToken = localStorage.getItem('token');
  if (savedToken) {
    token = savedToken;
    try {
      currentUser = await api('/auth/me');
      updateUserInfo();
    } catch {
      localStorage.removeItem('token');
      token = null;
      updateUserInfo();
    }
  } else {
    updateUserInfo();
  }
}

// Register service worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js').catch(() => { });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
