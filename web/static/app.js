/* ═══════════════════════════════════════════════════════════════
   FoodMind — Frontend Logic
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────
const queryInput     = document.getElementById('queryInput');
const searchBtn      = document.getElementById('searchBtn');
const statusDot      = document.getElementById('statusDot');
const statusLabel    = document.getElementById('statusLabel');

const pipelineSection = document.getElementById('pipelineSection');
const prefsSection    = document.getElementById('prefsSection');
const resultsSection  = document.getElementById('resultsSection');
const logsSection     = document.getElementById('logsSection');

const prefsGrid  = document.getElementById('prefsGrid');
const resultsGrid = document.getElementById('resultsGrid');
const resultsMeta = document.getElementById('resultsMeta');

const logsToggle = document.getElementById('logsToggle');
const logsBody   = document.getElementById('logsBody');
const logsPre    = document.getElementById('logsPre');
const toggleArrow = document.getElementById('toggleArrow');

const modalOverlay = document.getElementById('modalOverlay');
const modalClose   = document.getElementById('modalClose');
const modalContent = document.getElementById('modalContent');

const toast    = document.getElementById('toast');
const toastMsg = document.getElementById('toastMsg');

// ── Particle Canvas ───────────────────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById('particleCanvas');
  const ctx    = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x  = Math.random() * W;
      this.y  = Math.random() * H;
      this.vx = (Math.random() - 0.5) * 0.4;
      this.vy = (Math.random() - 0.5) * 0.4;
      this.r  = Math.random() * 1.5 + 0.5;
      this.alpha = Math.random() * 0.5 + 0.2;
      const colors = ['#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#c77dff'];
      this.color = colors[Math.floor(Math.random() * colors.length)];
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      if (this.x < 0 || this.x > W || this.y < 0 || this.y > H) this.reset();
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.globalAlpha = this.alpha;
      ctx.fill();
    }
  }

  for (let i = 0; i < 120; i++) particles.push(new Particle());

  // Draw connections
  function drawLines() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 100) {
          ctx.globalAlpha = (1 - dist / 100) * 0.08;
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth   = 0.5;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    ctx.globalAlpha = 1;
    particles.forEach(p => { p.update(); p.draw(); });
    drawLines();
    requestAnimationFrame(loop);
  }
  loop();
})();

// ── System Status ─────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    if (data.ollama && data.dataset) {
      statusDot.className   = 'status-dot online';
      statusLabel.textContent = `✓ System ready · ${data.dataset_rows.toLocaleString()} meals · ${data.model}`;
    } else if (data.dataset && !data.ollama) {
      statusDot.className   = 'status-dot partial';
      statusLabel.textContent = `⚠ Ollama offline — using rule-based fallback · ${data.dataset_rows.toLocaleString()} meals`;
    } else {
      statusDot.className   = 'status-dot offline';
      statusLabel.textContent = '✗ Dataset not found — run data/preprocess.py first';
    }
  } catch {
    statusDot.className   = 'status-dot offline';
    statusLabel.textContent = '✗ Backend unreachable';
  }
}
checkStatus();

// ── Quick chips ───────────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    queryInput.value = chip.dataset.q;
    queryInput.focus();
    queryInput.dispatchEvent(new Event('input'));
  });
});

// ── Enter key support ─────────────────────────────────────────────
queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') searchBtn.click();
});

// ── Pipeline state machine ────────────────────────────────────────
const agentNames = ['PreferenceAnalyzerAgent','MenuFetcherAgent','NutritionAnalyzerAgent','RecommendationAgent'];

function setPipelineState(cardIdx, state) {
  const card = document.getElementById(`pcard-${cardIdx}`);
  if (!card) return;
  card.className = `p-card glass ${state}`;
  const stateEl  = card.querySelector('.p-state');
  stateEl.className = `p-state ${state}`;
  stateEl.textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

function resetPipeline() {
  for (let i = 0; i < 4; i++) setPipelineState(i, 'idle');
  for (let i = 1; i <= 3; i++) {
    const c = document.getElementById(`connector${i}`);
    if (c) c.classList.remove('active');
  }
}

function animatePipeline(pipelineData) {
  return new Promise(resolve => {
    let i = 0;
    function next() {
      if (i >= pipelineData.length) { resolve(); return; }
      setPipelineState(i, 'running');
      const delay = Math.min(pipelineData[i].duration_ms, 1200);
      setTimeout(() => {
        const st = pipelineData[i].status === 'error' ? 'error' : 'done';
        setPipelineState(i, st);
        // activate connector
        if (i < 3) {
          const conn = document.getElementById(`connector${i + 1}`);
          if (conn) conn.classList.add('active');
        }
        i++;
        setTimeout(next, 180);
      }, Math.max(delay, 300));
    }
    next();
  });
}

// ── Preference summary ────────────────────────────────────────────
const PREF_META = {
  diet:          { icon: '🥗', label: 'Diet Type' },
  calorie_limit: { icon: '🔥', label: 'Calorie Limit' },
  exclude:       { icon: '🚫', label: 'Exclusions' },
  cuisine:       { icon: '🌍', label: 'Cuisine' }
};

function renderPreferences(prefs) {
  prefsGrid.innerHTML = '';
  for (const [key, meta] of Object.entries(PREF_META)) {
    let val = prefs[key];
    if (val === null || val === undefined || val === '') val = '—';
    else if (Array.isArray(val)) val = val.length ? val.join(', ') : '—';
    else if (key === 'calorie_limit') val = val === 9999 ? 'No limit' : `${val} kcal`;

    const card = document.createElement('div');
    card.className = 'pref-card';
    card.style.animationDelay = `${Object.keys(PREF_META).indexOf(key) * 0.08}s`;
    card.innerHTML = `
      <div class="pref-icon">${meta.icon}</div>
      <div class="pref-label">${meta.label}</div>
      <div class="pref-value">${val}</div>
    `;
    prefsGrid.appendChild(card);
  }
  show(prefsSection);
}

// ── Food emoji picker ─────────────────────────────────────────────
const FOOD_EMOJIS = ['🍲','🥘','🍛','🥗','🍜','🍱','🌮','🍝','🥙','🍣',
                     '🥩','🫕','🥦','🍚','🥐','🫔','🍤','🌯','🥞','🍗'];
function foodEmoji(name) {
  const seed = name.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  return FOOD_EMOJIS[seed % FOOD_EMOJIS.length];
}

// ── Meal Cards ────────────────────────────────────────────────────
function rankClass(i) {
  return i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : i === 2 ? 'rank-3' : 'rank-n';
}
function rankLabel(i) {
  return i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`;
}

function renderResults(recommendations, query) {
  resultsGrid.innerHTML = '';
  resultsMeta.textContent = `${recommendations.length} meal${recommendations.length !== 1 ? 's' : ''} found for "${query}"`;

  recommendations.forEach((meal, i) => {
    const card = document.createElement('div');
    card.className = 'meal-card';
    card.style.animationDelay = `${i * 0.08}s`;
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.id = `meal-card-${i}`;

    const tags = buildTags(meal);
    const scoreWidth = Math.round(meal.score * 100);

    card.innerHTML = `
      <div class="meal-rank ${rankClass(i)}">${rankLabel(i)}</div>
      <span class="meal-emoji">${foodEmoji(meal.name)}</span>
      <div class="meal-name">${escHtml(meal.name)}</div>
      <div class="meal-tags">${tags}</div>
      <div class="meal-macros">
        <div class="macro">
          <span class="macro-value macro-cal">${meal.calories}</span>
          <span class="macro-label">kcal</span>
        </div>
        <div class="macro">
          <span class="macro-value macro-pro">${meal.protein}g</span>
          <span class="macro-label">Protein</span>
        </div>
        <div class="macro">
          <span class="macro-value macro-fat">${meal.fat}g</span>
          <span class="macro-label">Fat</span>
        </div>
        <div class="macro">
          <span class="macro-value macro-car">${meal.carbs}g</span>
          <span class="macro-label">Carbs</span>
        </div>
      </div>
      <div class="meal-score-bar">
        <div class="meal-score-fill" data-width="${scoreWidth}"></div>
      </div>
    `;

    card.addEventListener('click', () => openMealModal(meal, i));
    card.addEventListener('keydown', e => { if (e.key === 'Enter') openMealModal(meal, i); });
    resultsGrid.appendChild(card);
  });

  show(resultsSection);

  // Animate score bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.meal-score-fill').forEach(el => {
      el.style.width = el.dataset.width + '%';
    });
  });
}

function buildTags(meal) {
  let html = '';
  if (meal.diet_type) html += `<span class="tag tag-diet">🥗 ${escHtml(meal.diet_type)}</span>`;
  if (meal.cuisine)   html += `<span class="tag tag-cuisine">🌍 ${escHtml(meal.cuisine)}</span>`;
  if (meal.allergens && meal.allergens !== 'none')
    html += `<span class="tag tag-allergy">⚠️ ${escHtml(meal.allergens)}</span>`;
  return html || '<span class="tag tag-diet">✓ Any diet</span>';
}

// ── Meal Modal ────────────────────────────────────────────────────
function openMealModal(meal, idx) {
  const tags = buildTags(meal);
  modalContent.innerHTML = `
    <div style="text-align:center; margin-bottom:20px;">
      <span style="font-size:4rem;">${foodEmoji(meal.name)}</span>
      <div style="font-family:var(--font-display);font-size:1.5rem;font-weight:700;margin-top:10px;line-height:1.3">
        ${escHtml(meal.name)}
      </div>
      <div class="meal-tags" style="justify-content:center;margin-top:10px;">${tags}</div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px;">
      ${macroBig('🔥', meal.calories, 'kcal', '#ff6b6b')}
      ${macroBig('💪', meal.protein + 'g', 'Protein', '#4d96ff')}
      ${macroBig('🧈', meal.fat + 'g', 'Fat', '#ffd93d')}
      ${macroBig('🌾', meal.carbs + 'g', 'Carbs', '#6bcb77')}
    </div>

    <div style="background:rgba(0,0,0,0.3);border-radius:16px;padding:16px;margin-bottom:16px;">
      <div style="font-size:0.78rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Ingredients</div>
      <div style="font-size:0.9rem;line-height:1.7;color:var(--text);">${escHtml(meal.ingredients || 'Not available')}</div>
    </div>

    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div style="font-size:0.85rem;color:var(--text-muted);">MAS Nutrition Score</div>
      <div style="font-size:1.1rem;font-weight:700;background:var(--grad-btn);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
        ${Math.round(meal.score * 100)}%
      </div>
    </div>
    <div class="meal-score-bar" style="margin-top:8px;">
      <div class="meal-score-fill" data-width="${Math.round(meal.score * 100)}"></div>
    </div>
  `;

  modalOverlay.classList.remove('hidden');
  document.body.style.overflow = 'hidden';

  requestAnimationFrame(() => {
    modalContent.querySelectorAll('.meal-score-fill').forEach(el => {
      el.style.width = el.dataset.width + '%';
    });
  });
}

function macroBig(icon, val, label, color) {
  return `
    <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:14px;text-align:center;">
      <div style="font-size:1.4rem;">${icon}</div>
      <div style="font-size:1.1rem;font-weight:700;color:${color};margin:4px 0;">${val}</div>
      <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em">${label}</div>
    </div>`;
}

modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

function closeModal() {
  modalOverlay.classList.add('hidden');
  document.body.style.overflow = '';
}

// ── Agent Logs ────────────────────────────────────────────────────
function renderLogs(logs) {
  if (!logs || !logs.length) return;
  logsPre.textContent = logs.map(log => {
    const out = JSON.stringify(log.output, null, 2);
    return `[${log.timestamp}] ${log.agent}\n  INPUT:  ${JSON.stringify(log.input)}\n  OUTPUT: ${out}\n`;
  }).join('\n─────────────────────────────────────────\n\n');
  show(logsSection);
}

logsToggle.addEventListener('click', () => {
  const isOpen = !logsBody.classList.contains('hidden');
  logsBody.classList.toggle('hidden', isOpen);
  toggleArrow.classList.toggle('open', !isOpen);
});

// ── Main search ───────────────────────────────────────────────────
searchBtn.addEventListener('click', async () => {
  const query = queryInput.value.trim();
  if (!query) {
    showToast('⚠️ Please enter your food preference first');
    queryInput.focus();
    return;
  }

  // Reset UI
  hide(prefsSection);
  hide(resultsSection);
  hide(logsSection);
  resetPipeline();
  show(pipelineSection);
  pipelineSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Loading state
  searchBtn.disabled = true;
  searchBtn.classList.add('loading');
  searchBtn.innerHTML = `<div class="spinner"></div><span class="btn-text">Analysing…</span>`;

  try {
    const response = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });

    const data = await response.json();

    if (!response.ok) {
      showToast(`❌ ${data.error || 'Something went wrong'}`);
      resetBtn();
      return;
    }

    // Animate pipeline
    await animatePipeline(data.pipeline);

    // Show results
    renderPreferences(data.preferences);
    renderResults(data.recommendations, data.query);
    renderLogs(data.logs);

    // Scroll to results
    setTimeout(() => {
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 400);

    if (data.recommendations.length === 0) {
      showToast('🔍 No meals matched your criteria. Try broadening your preferences.');
    } else {
      showToast(`✅ Found ${data.recommendations.length} perfect meals for you!`);
    }

    // Refresh status after query
    checkStatus();

  } catch (err) {
    showToast('❌ Cannot reach the backend. Is the Flask server running?');
    console.error(err);
    resetPipeline();
  } finally {
    resetBtn();
  }
});

function resetBtn() {
  searchBtn.disabled = false;
  searchBtn.classList.remove('loading');
  searchBtn.innerHTML = `<span class="btn-text">Find Meals</span><span class="btn-icon">→</span>`;
}

// ── Helpers ───────────────────────────────────────────────────────
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }
function escHtml(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

let toastTimer;
function showToast(msg) {
  toastMsg.textContent = msg;
  toast.classList.remove('hidden');
  requestAnimationFrame(() => toast.classList.add('show'));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.classList.add('hidden'), 400);
  }, 3500);
}

// ── Navbar scroll effect ──────────────────────────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('navbar').style.background =
    window.scrollY > 40 ? 'rgba(8,11,20,0.92)' : 'rgba(8,11,20,0.7)';
});
