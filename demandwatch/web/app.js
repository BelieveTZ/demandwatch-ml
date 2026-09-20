/* DemandWatch ML — frontend controller.
   Reads ./data/report.json (produced by the offline pipeline) and renders the
   backtest dashboard. Optionally talks to a local FastAPI service at
   /api/health and /api/predict. No frameworks, no network dependencies. */

'use strict';

(function () {
  var REPORT_URL = './data/report.json';
  var HEALTH_URL = '/api/health';
  var PREDICT_URL = '/api/predict';
  var SVG_NS = 'http://www.w3.org/2000/svg';

  var FEATURE_META = [
    { key: 'hour', label: 'Hour of day', meaning: 'Target hour, 0–23', digits: 0, editable: false },
    { key: 'weekday', label: 'Weekday', meaning: '0 = Monday … 6 = Sunday', digits: 0, editable: false },
    { key: 'month', label: 'Month', meaning: 'Calendar month, 1–12', digits: 0, editable: false },
    { key: 'workingday', label: 'Working day', meaning: '1 = neither weekend nor holiday', digits: 0, editable: false },
    { key: 'holiday', label: 'Holiday', meaning: '1 = US federal holiday', digits: 0, editable: false },
    { key: 'trend_days', label: 'Trend', meaning: 'Days since 2011-01-01', digits: 0, editable: false },
    { key: 'lag_1h', label: 'Demand, 1 h earlier', meaning: 'Observed count at t − 1 h', digits: 0, editable: true, min: 0, step: 1 },
    { key: 'lag_24h', label: 'Demand, 24 h earlier', meaning: 'Observed count at t − 24 h', digits: 0, editable: true, min: 0, step: 1 },
    { key: 'lag_168h', label: 'Demand, 168 h earlier', meaning: 'Observed count at t − 168 h', digits: 0, editable: true, min: 0, step: 1 },
    { key: 'mean_24h', label: 'Rolling mean, 24 h', meaning: 'Mean demand over the previous 24 h', digits: 1, editable: false },
    { key: 'mean_168h', label: 'Rolling mean, 168 h', meaning: 'Mean demand over the previous 168 h', digits: 1, editable: false },
    { key: 'temp_previous', label: 'Temperature, previous hour', meaning: 'Normalized 0–1, at t − 1 h', digits: 3, editable: true, min: 0, max: 1, step: 0.01 },
    { key: 'humidity_previous', label: 'Humidity, previous hour', meaning: 'Normalized 0–1, at t − 1 h', digits: 3, editable: true, min: 0, max: 1, step: 0.01 },
    { key: 'wind_previous', label: 'Wind speed, previous hour', meaning: 'Normalized 0–1, at t − 1 h', digits: 3, editable: false },
    { key: 'weather_previous', label: 'Weather category, previous hour', meaning: '1 clear … 4 heavy rain or snow, at t − 1 h', digits: 0, editable: false }
  ];

  var SPLIT_USE = {
    train: 'Fit candidate models',
    validation: 'Select configuration',
    calibration: 'Fit interval radius',
    test: 'Final evaluation only'
  };

  var NF0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
  var NF1 = new Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  var NF2 = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  var NF3 = new Intl.NumberFormat('en-US', { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  var DT_LONG = new Intl.DateTimeFormat('en-US', { timeZone: 'UTC', month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
  var DT_HOUR = new Intl.DateTimeFormat('en-US', { timeZone: 'UTC', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
  var DT_DAY = new Intl.DateTimeFormat('en-US', { timeZone: 'UTC', month: 'short', day: 'numeric' });
  var DT_DAY_FULL = new Intl.DateTimeFormat('en-US', { timeZone: 'UTC', weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
  var DT_STAMP = new Intl.DateTimeFormat('en-US', { timeZone: 'UTC', month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });

  var state = {
    report: null,
    points: [],
    examples: [],
    exampleIndex: 0,
    granularity: 'week',
    periodKey: '',
    showSeasonal: true,
    monitorWindow: 'observed',
    health: 'checking',
    healthDetail: '',
    live: null,
    liveEpoch: 0,
    liveError: '',
    liveNotice: '',
    busy: false,
    lastChartWidth: 0
  };

  /* ------------------------------- helpers ------------------------------- */

  function $(id) { return document.getElementById(id); }

  function make(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function svgEl(name, attrs) {
    var node = document.createElementNS(SVG_NS, name);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) { node.setAttribute(key, String(attrs[key])); });
    }
    return node;
  }

  function svgText(x, y, value, anchor) {
    var node = svgEl('text', { x: x, y: y, class: 'axis-text', 'text-anchor': anchor || 'middle' });
    node.textContent = value;
    return node;
  }

  function setText(id, value) {
    var node = $(id);
    if (node) node.textContent = value === undefined || value === null ? '—' : String(value);
  }

  function show(node, visible) {
    if (!node) return;
    if (visible) node.removeAttribute('hidden');
    else node.setAttribute('hidden', '');
  }

  function isNum(value) { return typeof value === 'number' && isFinite(value); }

  function fmt(value, digits) {
    if (!isNum(value)) return '—';
    if (digits === 0) return NF0.format(value);
    if (digits === 1) return NF1.format(value);
    if (digits === 3) return NF3.format(value);
    return NF2.format(value);
  }

  function pct(value, digits) {
    if (!isNum(value)) return '—';
    return (digits === 0 ? NF0.format(value * 100) : NF1.format(value * 100)) + '%';
  }

  function signed(value, digits) {
    if (!isNum(value)) return '—';
    return (value > 0 ? '+' : value < 0 ? '−' : '') + fmt(Math.abs(value), digits);
  }

  function parseTs(value) {
    if (typeof value !== 'string' || !value) return NaN;
    var text = value.trim().replace(' ', 'T');
    // Preserve naive source wall-clock labels in every visitor's timezone.
    if (!/(Z|[+-]\d{2}:\d{2})$/.test(text)) text += 'Z';
    var ms = Date.parse(text);
    if (isFinite(ms)) return ms;
    ms = Date.parse(text + 'Z');
    return isFinite(ms) ? ms : NaN;
  }

  function pad2(value) { return value < 10 ? '0' + value : String(value); }

  function dayKey(ms) {
    var d = new Date(ms);
    return d.getUTCFullYear() + '-' + pad2(d.getUTCMonth() + 1) + '-' + pad2(d.getUTCDate());
  }

  function weekStartMs(ms) {
    var d = new Date(ms);
    var offset = (d.getUTCDay() + 6) % 7;
    d.setUTCHours(0, 0, 0, 0);
    d.setUTCDate(d.getUTCDate() - offset);
    return d.getTime();
  }

  function shorten(value, max) {
    var text = String(value === undefined || value === null ? '' : value);
    return text.length > max ? text.slice(0, max) + '…' : text;
  }

  /* -------------------------------- state -------------------------------- */

  function setStatus(mode, detail) {
    var region = $('app-status');
    if (!region) return;
    region.replaceChildren();

    if (mode === 'loading') {
      var loading = make('div', 'notice notice-loading');
      loading.appendChild(make('p', 'notice-title', 'Loading backtest report…'));
      loading.appendChild(make('p', 'notice-body', 'Requesting ' + REPORT_URL));
      region.appendChild(loading);
      return;
    }

    if (mode === 'error') {
      var card = make('div', 'notice notice-error');
      card.appendChild(make('p', 'notice-title', 'Report data could not be loaded'));
      card.appendChild(make('p', 'notice-body',
        'The dashboard requested ' + REPORT_URL + ' and it did not load: ' + (detail || 'unknown error') + '.'));
      var steps = make('ol', 'notice-steps');
      [
        'Run the project pipeline to generate the evaluation report JSON.',
        'Place the file at demandwatch/web/data/report.json so it sits next to this page.',
        'Serve this folder over HTTP, for example: python -m http.server 8000 (run inside demandwatch/web). Opening index.html from disk blocks fetch in most browsers.',
        'Reload this page, or use Retry below.'
      ].forEach(function (line) { steps.appendChild(make('li', null, line)); });
      card.appendChild(steps);
      var actions = make('div', 'notice-actions');
      var retry = make('button', 'btn btn-primary', 'Retry');
      retry.type = 'button';
      retry.addEventListener('click', function () { loadReport(); });
      var openApi = make('a', 'btn btn-ghost', 'Start the local API');
      openApi.href = 'https://github.com/BelieveTZ/demandwatch-ml';
      openApi.target = '_blank';
      openApi.rel = 'noopener noreferrer';
      actions.appendChild(retry);
      actions.appendChild(openApi);
      card.appendChild(actions);
      region.appendChild(card);
      return;
    }

    region.replaceChildren();
  }

  /* --------------------------------- boot -------------------------------- */

  function init() {
    wireTabs();
    wireOverview();
    wireLab();
    wireMonitoring();
    loadReport();
    probeHealth();
  }

  function loadReport() {
    setStatus('loading', '');
    var request;
    try {
      request = fetch(REPORT_URL, { cache: 'no-store', headers: { Accept: 'application/json' } });
    } catch (err) {
      fail(err);
      return;
    }
    request.then(function (response) {
      if (!response.ok) throw new Error('HTTP ' + response.status + ' ' + (response.statusText || ''));
      return response.text();
    }).then(function (text) {
      var data;
      try {
        data = JSON.parse(text);
      } catch (err) {
        throw new Error('the response is not valid JSON');
      }
      var missing = ['run_id', 'dataset', 'splits', 'metrics', 'selected_model', 'intervals',
        'validation', 'importance', 'slices', 'series', 'examples', 'monitoring', 'benchmark']
        .filter(function (key) { return data[key] === undefined || data[key] === null; });
      if (missing.length) throw new Error('the report is missing required fields: ' + missing.join(', '));
      if (!Array.isArray(data.series) || !data.series.length) throw new Error('the report has an empty series array');
      accept(data);
    }).catch(fail);
  }

  function fail(err) {
    var message = err && err.message ? err.message : 'unknown error';
    if (/failed to fetch|networkerror|load failed/i.test(message)) {
      message = 'the request failed (offline, wrong path, or the page was opened directly from disk)';
    }
    state.report = null;
    ['overview', 'lab', 'monitoring', 'methods'].forEach(function (name) {
      var panel = $('panel-' + name);
      if (panel) panel.setAttribute('hidden', '');
    });
    setStatus('error', message);
  }

  function accept(report) {
    state.report = report;
    state.points = report.series.map(function (row) {
      return {
        t: parseTs(row.timestamp),
        actual: isNum(row.actual) ? row.actual : null,
        prediction: isNum(row.prediction) ? row.prediction : null,
        lower: isNum(row.lower) ? row.lower : null,
        upper: isNum(row.upper) ? row.upper : null,
        seasonal: isNum(row.seasonal) ? row.seasonal : null,
        timestamp: row.timestamp
      };
    }).filter(function (point) { return isFinite(point.t); })
      .sort(function (a, b) { return a.t - b.t; });

    state.examples = Array.isArray(report.examples) ? report.examples.slice() : [];
    state.exampleIndex = 0;
    state.granularity = 'week';
    $('range-granularity').value = 'week';
    state.periodKey = '';
    state.live = null;
    state.liveError = '';

    setText('meta-run-id', shorten(report.run_id, 28));
    setText('meta-created', isNum(parseTs(report.created_at)) ? DT_STAMP.format(new Date(parseTs(report.created_at))) : String(report.created_at || '—'));
    var runIdNode = $('meta-run-id');
    if (runIdNode) runIdNode.title = String(report.run_id || '');

    setStatus('ready', '');
    fillPeriodSelect();
    activateTab('overview', false);
    show($('panel-overview'), true);

    renderOverview();
    renderLab();
    renderMonitoring();
    renderMethods();
  }

  /* --------------------------------- tabs -------------------------------- */

  function wireTabs() {
    var buttons = Array.prototype.slice.call(document.querySelectorAll('.nav-item[role="tab"]'));
    buttons.forEach(function (button, index) {
      button.addEventListener('click', function () { activateTab(button.dataset.panel, true); });
      button.addEventListener('keydown', function (event) {
        var delta = event.key === 'ArrowDown' || event.key === 'ArrowRight' ? 1
          : event.key === 'ArrowUp' || event.key === 'ArrowLeft' ? -1 : 0;
        if (!delta) return;
        event.preventDefault();
        var next = buttons[(index + delta + buttons.length) % buttons.length];
        next.focus();
        activateTab(next.dataset.panel, true);
      });
    });
    document.addEventListener('click', function (event) {
      var target = event.target.closest ? event.target.closest('[data-goto]') : null;
      if (target) activateTab(target.dataset.goto, true);
    });
  }

  function activateTab(name, moveFocus) {
    if (!name) return;
    var buttons = Array.prototype.slice.call(document.querySelectorAll('.nav-item[role="tab"]'));
    buttons.forEach(function (button) {
      var active = button.dataset.panel === name;
      button.setAttribute('aria-selected', active ? 'true' : 'false');
      button.tabIndex = active ? 0 : -1;
      if (active && moveFocus) button.focus();
    });
    ['overview', 'lab', 'monitoring', 'methods'].forEach(function (panelName) {
      var panel = $('panel-' + panelName);
      if (panel) show(panel, panelName === name);
    });
    if (name === 'overview' && state.report) drawTimeline();
  }

  /* ------------------------------- overview ------------------------------ */

  function wireOverview() {
    var download = $('download-report');
    if (download) download.addEventListener('click', downloadReport);

    var granularity = $('range-granularity');
    if (granularity) {
      granularity.addEventListener('change', function () {
        state.granularity = granularity.value;
        state.periodKey = '';
        fillPeriodSelect();
        drawTimeline();
      });
    }
    var period = $('range-value');
    if (period) {
      period.addEventListener('change', function () {
        state.periodKey = period.value;
        drawTimeline();
      });
    }
    var seasonal = $('show-seasonal');
    if (seasonal) {
      seasonal.addEventListener('change', function () {
        state.showSeasonal = seasonal.checked;
        drawTimeline();
      });
    }

    var host = $('chart-host');
    if (host && window.ResizeObserver) {
      var observer = new ResizeObserver(function () {
        if (!state.report) return;
        var width = host.clientWidth;
        if (Math.abs(width - state.lastChartWidth) < 2) return;
        drawTimeline();
      });
      observer.observe(host);
    } else {
      window.addEventListener('resize', function () {
        if (state.report) drawTimeline();
      });
    }
  }

  function downloadReport() {
    if (!state.report) return;
    var label = $('download-label');
    var name = 'demandwatch-report-' + String(state.report.run_id || 'run').replace(/[^A-Za-z0-9._-]/g, '_') + '.json';
    var blob = new Blob([JSON.stringify(state.report, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 3000);
    if (label) {
      var original = label.textContent;
      label.textContent = 'Report downloaded';
      window.setTimeout(function () { label.textContent = original; }, 1800);
    }
  }

  function renderOverview() {
    var report = state.report;
    var metrics = Array.isArray(report.metrics) ? report.metrics.slice() : [];
    var intervals = report.intervals || {};
    var benchmark = report.benchmark || {};
    var selected = String(report.selected_model || '');
    var selectedMetric = null;
    var seasonalMetric = null;

    metrics.forEach(function (metric) {
      if (metric.model === selected) selectedMetric = metric;
      if (/seasonal\s*168/i.test(String(metric.model))) seasonalMetric = metric;
    });

    /* top stats */
    setText('stat-mae-label', 'Test MAE · ' + (selected || 'selected model'));
    setText('stat-mae-value', selectedMetric ? fmt(selectedMetric.mae, 2) : '—');
    if (selectedMetric && seasonalMetric && isNum(seasonalMetric.mae) && seasonalMetric.mae !== 0) {
      var diff = selectedMetric.mae - seasonalMetric.mae;
      var rel = (diff / seasonalMetric.mae) * 100;
      setText('stat-mae-sub', 'rides/hour · ' + signed(rel, 1) + '% vs Seasonal 168h (' + signed(diff, 2) + ')');
    } else {
      setText('stat-mae-sub', selectedMetric ? 'rides/hour mean absolute error' : 'not reported');
    }

    setText('stat-cov-value', isNum(intervals.coverage) ? pct(intervals.coverage, 1) : '—');
    var coverageNote = '';
    if (isNum(intervals.coverage) && isNum(intervals.target)) {
      var gap = intervals.coverage - intervals.target;
      coverageNote = Math.abs(gap) <= 0.02 ? ' · on target' : (gap > 0 ? ' · above target' : ' · below target');
    }
    setText('stat-cov-sub', 'target ' + (isNum(intervals.target) ? pct(intervals.target, 0) : '—')
      + ' · mean total width ' + fmt(intervals.mean_width, 0) + ' rides'
      + (isNum(intervals.n) ? ' · n ' + fmt(intervals.n, 0) : '') + coverageNote);

    setText('stat-lat-value', isNum(benchmark.p95_ms) ? fmt(benchmark.p95_ms, 1) + ' ms' : '—');
    setText('stat-lat-sub', (isNum(benchmark.n) ? 'p95 over ' + fmt(benchmark.n, 0) + ' local calls · ' : '')
      + 'CPU only, hardware-dependent');
    var latencyNode = $('stat-lat-value');
    if (latencyNode) {
      var benchmarkScope = [benchmark.scope, benchmark.platform, benchmark.processor].filter(function (item) {
        return typeof item === 'string' && item;
      });
      if (benchmarkScope.length) latencyNode.title = benchmarkScope.join(' · ');
    }

    setText('stat-model-value', selected || '—');
    var validation = Array.isArray(report.validation) ? report.validation : [];
    var selectedValidation = validation.filter(function (row) { return row.model === selected; })
      .sort(function (a, b) { return a.mae - b.mae; })[0];
    setText('stat-model-sub', 'chosen on validation MAE'
      + (selectedValidation ? ' (' + fmt(selectedValidation.mae, 2) + ')' : '')
      + ' · ' + validation.length + ' candidates compared');

    /* chart labels */
    setText('legend-model', selected || 'selected model');
    setText('legend-interval', (isNum(intervals.target) ? pct(intervals.target, 0) : '90%') + ' interval');
    var target = isNum(intervals.target) ? intervals.target : 0.9;
    setText('chart-meta', state.points.length
      ? state.points.length.toLocaleString('en-US') + ' hourly test points · '
        + DT_LONG.format(new Date(state.points[0].t)) + ' → ' + DT_LONG.format(new Date(state.points[state.points.length - 1].t))
        + ' · ' + (isNum(intervals.target) ? pct(intervals.target, 0) : '90%') + ' calibrated interval'
      : 'No series points in the report.');
    setText('models-sub', 'Test block · ' + fmt((selectedMetric || metrics[0] || {}).n, 0)
      + ' eligible hours per model · chronological split, no shuffling.');

    /* model table */
    var modelBody = $('model-table-body');
    if (modelBody) {
      modelBody.replaceChildren();
      var sorted = metrics.slice().sort(function (a, b) { return (a.mae || 0) - (b.mae || 0); });
      var best = sorted.length ? sorted[0].mae : null;
      if (!sorted.length) {
        modelBody.appendChild(emptyRow(5, 'No model metrics were reported.'));
      }
      sorted.forEach(function (metric) {
        var row = document.createElement('tr');
        if (metric.model === selected) row.className = 'is-selected';
        var name = make('th', null, String(metric.model));
        name.scope = 'row';
        if (metric.model === selected) name.appendChild(make('span', 'row-tag', 'selected'));
        row.appendChild(name);
        row.appendChild(numCell(fmt(metric.n, 0)));
        row.appendChild(numCell(fmt(metric.mae, 2)));
        row.appendChild(numCell(fmt(metric.rmse, 2)));
        var relative = isNum(metric.mae) && isNum(best) && best > 0 ? (metric.mae / best - 1) * 100 : null;
        row.appendChild(numCell(relative === null ? '—' : (relative < 0.05 ? 'best' : '+' + NF1.format(relative) + '%')));
        modelBody.appendChild(row);
      });
    }

    /* importance */
    var importanceHost = $('importance-chart');
    var importance = Array.isArray(report.importance) ? report.importance.slice() : [];
    if (importanceHost) {
      importanceHost.replaceChildren();
      if (!importance.length) {
        importanceHost.appendChild(make('p', 'muted small', 'No permutation importance was reported.'));
      } else {
        var maxIncrease = importance.reduce(function (acc, row) {
          return Math.max(acc, isNum(row.mae_increase) ? row.mae_increase : 0);
        }, 0);
        importance.forEach(function (row) {
          importanceHost.appendChild(barRow({
            label: String(row.feature),
            ratio: maxIncrease > 0 ? (isNum(row.mae_increase) ? row.mae_increase : 0) / maxIncrease : 0,
            value: '+' + fmt(row.mae_increase, 2) + (isNum(row.std) ? ' ± ' + fmt(row.std, 2) : ''),
            alert: false
          }));
        });
        setText('importance-sub', 'Train-only model · validation permutation importance · ' + importance.length
          + ' features · largest increase +' + fmt(maxIncrease, 2) + ' MAE');
      }
    }

    /* slices */
    var sliceBody = $('slice-table-body');
    var slices = Array.isArray(report.slices) ? report.slices : [];
    if (sliceBody) {
      sliceBody.replaceChildren();
      if (!slices.length) sliceBody.appendChild(emptyRow(5, 'No slice breakdown was reported.'));
      slices.forEach(function (slice) {
        var row = document.createElement('tr');
        var name = make('th', null, String(slice.slice));
        name.scope = 'row';
        row.appendChild(name);
        row.appendChild(numCell(fmt(slice.n, 0)));
        row.appendChild(numCell(fmt(slice.mae, 2)));
        row.appendChild(numCell(fmt(slice.rmse, 2)));
        var cell = make('td', 'num');
        cell.appendChild(make('span', null, isNum(slice.coverage) ? pct(slice.coverage, 1) : '—'));
        if (isNum(slice.coverage) && isNum(intervals.target) && slice.coverage < intervals.target - 0.02) {
          cell.appendChild(make('span', 'row-tag', 'under'));
          row.className = 'is-alert';
        }
        row.appendChild(cell);
        sliceBody.appendChild(row);
      });
    }

    /* caveats */
    var caveats = Array.isArray(report.caveats) ? report.caveats : [];
    fillCaveats($('overview-caveats'), caveats.slice(0, 4), caveats.length > 4);

    /* side card */
    setText('side-model-name', selected || 'Not reported');
    var sideNote = selectedMetric
      ? 'Test MAE ' + fmt(selectedMetric.mae, 2) + ' rides/hour over ' + fmt(selectedMetric.n, 0)
        + ' hours · RMSE ' + fmt(selectedMetric.rmse, 2) + '.'
      : 'The report does not carry a metric row for the selected model.';
    setText('side-model-note', sideNote);
    fillFacts($('run-facts'), runFacts());
  }

  function emptyRow(span, text) {
    var row = document.createElement('tr');
    var cell = make('td', 'muted', text);
    cell.colSpan = span;
    row.appendChild(cell);
    return row;
  }

  function numCell(text) {
    var cell = make('td', 'num', text);
    return cell;
  }

  function barRow(options) {
    var row = make('div', 'bar-row');
    row.appendChild(make('span', 'bar-label', options.label));
    var track = make('div', 'bar-track');
    var fill = make('div', 'bar-fill');
    var ratio = Math.max(0, Math.min(1, options.ratio || 0));
    fill.style.width = (ratio * 100).toFixed(2) + '%';
    if (!ratio) fill.className = 'bar-fill is-zero';
    if (options.alert) fill.className = 'bar-fill is-alert';
    track.appendChild(fill);
    if (isNum(options.thresholdRatio)) {
      var mark = make('div', 'psi-threshold');
      mark.style.left = (Math.max(0, Math.min(1, options.thresholdRatio)) * 100).toFixed(2) + '%';
      track.appendChild(mark);
    }
    row.appendChild(track);
    row.appendChild(make('span', 'bar-value', options.value));
    return row;
  }

  function fillFacts(host, pairs) {
    if (!host) return;
    host.replaceChildren();
    pairs.forEach(function (pair) {
      var wrap = make('div');
      wrap.appendChild(make('dt', null, pair[0]));
      var dd = make('dd', null, pair[1]);
      if (pair[2]) dd.title = pair[2];
      wrap.appendChild(dd);
      host.appendChild(wrap);
    });
  }

  function fillCaveats(host, caveats, more) {
    if (!host) return;
    host.replaceChildren();
    if (!caveats.length) {
      host.appendChild(make('li', 'muted', 'The report recorded no caveats.'));
      return;
    }
    caveats.forEach(function (text) { host.appendChild(make('li', null, String(text))); });
    if (more) {
      host.appendChild(make('li', 'muted', 'Additional caveats are listed in Methods.'));
    }
  }

  function runFacts() {
    var report = state.report;
    var data = report.dataset || {};
    var splits = report.splits || {};
    var test = splits.test || {};
    var calibration = splits.calibration || {};
    var facts = [
      ['Eligible rows', fmt(data.eligible_rows, 0)],
      ['Source rows', fmt(data.raw_rows, 0)],
      ['Missing hours', fmt(data.missing_hours, 0)],
      ['Coverage', String(data.start || '—').slice(0, 10) + ' → ' + String(data.end || '—').slice(0, 10)],
      ['Test block', fmt(test.n, 0) + ' rows'],
      ['Test window', String(test.start || '—').slice(0, 10) + ' → ' + String(test.end || '—').slice(0, 10)],
      ['Calibration block', fmt(calibration.n, 0) + ' rows'],
      ['Interval radius', isNum((report.intervals || {}).radius) ? '±' + fmt(report.intervals.radius, 2) : '—']
    ];
    return facts;
  }

  /* -------------------------------- chart -------------------------------- */

  function periodKeyFor(ms, mode) {
    return mode === 'week' ? dayKey(weekStartMs(ms)) : dayKey(ms);
  }

  function seriesView() {
    if (state.granularity === 'all' || !state.periodKey) return state.points;
    return state.points.filter(function (point) {
      return periodKeyFor(point.t, state.granularity) === state.periodKey;
    });
  }

  function fillPeriodSelect() {
    var wrap = $('range-value-wrap');
    var select = $('range-value');
    if (!wrap || !select) return;
    if (state.granularity === 'all') {
      show(wrap, false);
      return;
    }
    show(wrap, true);
    var buckets = [];
    var index = {};
    state.points.forEach(function (point) {
      var key = periodKeyFor(point.t, state.granularity);
      if (!index[key]) {
        index[key] = { key: key, count: 0, first: point.t };
        buckets.push(index[key]);
      }
      index[key].count += 1;
    });
    select.replaceChildren();
    buckets.forEach(function (bucket) {
      var option = document.createElement('option');
      option.value = bucket.key;
      var label = state.granularity === 'week'
        ? 'Week of ' + DT_DAY_FULL.format(new Date(parseTs(bucket.key + 'T00:00:00')))
        : DT_DAY_FULL.format(new Date(parseTs(bucket.key + 'T00:00:00')));
      option.textContent = label + ' · ' + bucket.count + ' h';
      select.appendChild(option);
    });
    if (!buckets.length) {
      state.periodKey = '';
      return;
    }
    var keep = buckets.some(function (bucket) { return bucket.key === state.periodKey; });
    state.periodKey = keep ? state.periodKey : buckets[0].key;
    select.value = state.periodKey;
  }

  function currentRangeLabel() {
    if (state.granularity === 'all' || !state.periodKey) return 'full test window';
    var when = DT_DAY_FULL.format(new Date(parseTs(state.periodKey + 'T00:00:00')));
    return state.granularity === 'week' ? 'week of ' + when : when;
  }

  function niceMax(value) {
    if (!isFinite(value) || value <= 0) return 1;
    var exponent = Math.floor(Math.log10(value));
    var base = Math.pow(10, exponent);
    var steps = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
    for (var i = 0; i < steps.length; i += 1) {
      if (value <= steps[i] * base) return steps[i] * base;
    }
    return 10 * base;
  }

  function linePath(points, key, X, Y) {
    var d = '';
    var pen = false;
    points.forEach(function (point) {
      var value = point[key];
      if (!isNum(value)) { pen = false; return; }
      d += (pen ? ' L ' : ' M ') + X(point.t).toFixed(2) + ' ' + Y(value).toFixed(2);
      pen = true;
    });
    return d;
  }

  function bandPath(points, X, Y) {
    var d = '';
    var segment = [];
    function flush() {
      if (segment.length > 1) {
        d += ' M ' + segment.map(function (point) {
          return X(point.t).toFixed(2) + ' ' + Y(point.upper).toFixed(2);
        }).join(' L ');
        d += ' L ' + segment.slice().reverse().map(function (point) {
          return X(point.t).toFixed(2) + ' ' + Y(point.lower).toFixed(2);
        }).join(' L ') + ' Z';
      }
      segment = [];
    }
    points.forEach(function (point) {
      if (isNum(point.upper) && isNum(point.lower)) segment.push(point);
      else flush();
    });
    flush();
    return d;
  }

  function nearestIndex(points, t) {
    var lo = 0;
    var hi = points.length - 1;
    while (hi - lo > 1) {
      var mid = (lo + hi) >> 1;
      if (points[mid].t <= t) lo = mid; else hi = mid;
    }
    return (t - points[lo].t) <= (points[hi].t - t) ? lo : hi;
  }

  function drawTimeline() {
    var host = $('chart-host');
    var readout = $('chart-readout');
    if (!host || !state.report) return;
    var points = seriesView();
    var rangeLabel = currentRangeLabel();
    host.replaceChildren();

    if (!points.length) {
      host.appendChild(make('p', 'muted small', 'No series points in the selected range.'));
      setText('chart-summary', 'Chart summary: no points to display.');
      return;
    }

    var width = Math.max(320, Math.round(host.clientWidth || 860));
    state.lastChartWidth = host.clientWidth;
    var height = width < 620 ? 280 : 370;
    var margin = { top: 16, right: 18, bottom: 38, left: 62 };
    var innerWidth = Math.max(20, width - margin.left - margin.right);
    var innerHeight = Math.max(20, height - margin.top - margin.bottom);

    var t0 = points[0].t;
    var t1 = points[points.length - 1].t;
    var span = Math.max(1, t1 - t0);

    var peak = 0;
    points.forEach(function (point) {
      [point.actual, point.prediction, point.upper].forEach(function (value) {
        if (isNum(value)) peak = Math.max(peak, value);
      });
      if (state.showSeasonal && isNum(point.seasonal)) peak = Math.max(peak, point.seasonal);
    });
    var yTop = niceMax(peak * 1.08);

    function X(t) { return margin.left + ((t - t0) / span) * innerWidth; }
    function Y(value) { return margin.top + innerHeight - (Math.max(0, value) / yTop) * innerHeight; }

    var svg = svgEl('svg', {
      class: 'chart-svg',
      viewBox: '0 0 ' + width + ' ' + height,
      width: width,
      height: height,
      role: 'img',
      'aria-label': 'Time series of actual demand and model forecast with an interval band, ' + rangeLabel + '.'
    });

    for (var i = 0; i <= 4; i += 1) {
      var value = (yTop / 4) * i;
      var y = Y(value);
      svg.appendChild(svgEl('line', { class: 'grid-line', x1: margin.left, x2: margin.left + innerWidth, y1: y, y2: y }));
      svg.appendChild(svgText(margin.left - 8, y + 4, NF0.format(value), 'end'));
    }

    var tickCount = Math.max(2, Math.min(7, Math.floor(innerWidth / 118)));
    for (var j = 0; j < tickCount; j += 1) {
      var tickTime = t0 + (span * j) / (tickCount - 1);
      var label = span <= 3 * 86400000 ? DT_HOUR.format(new Date(tickTime)) : DT_DAY.format(new Date(tickTime));
      svg.appendChild(svgText(X(tickTime), margin.top + innerHeight + 20, label, 'middle'));
    }
    svg.appendChild(svgEl('line', {
      class: 'axis-line', x1: margin.left, x2: margin.left + innerWidth,
      y1: margin.top + innerHeight, y2: margin.top + innerHeight
    }));

    var band = bandPath(points, X, Y);
    if (band) svg.appendChild(svgEl('path', { class: 'band', d: band }));

    if (state.showSeasonal) {
      var seasonalPath = linePath(points, 'seasonal', X, Y);
      if (seasonalPath) svg.appendChild(svgEl('path', { class: 'line-seasonal', d: seasonalPath }));
    }
    var actualPath = linePath(points, 'actual', X, Y);
    if (actualPath) svg.appendChild(svgEl('path', { class: 'line-actual', d: actualPath }));
    var predictionPath = linePath(points, 'prediction', X, Y);
    if (predictionPath) svg.appendChild(svgEl('path', { class: 'line-pred', d: predictionPath }));

    var crosshair = svgEl('line', { class: 'crosshair', y1: margin.top, y2: margin.top + innerHeight, x1: 0, x2: 0, opacity: 0 });
    var dotActual = svgEl('circle', { r: 3.6, fill: '#13243a', stroke: '#fff', 'stroke-width': 1.5, opacity: 0 });
    var dotPred = svgEl('circle', { r: 3.6, fill: '#087f73', stroke: '#fff', 'stroke-width': 1.5, opacity: 0 });
    var dotSeasonal = svgEl('circle', { r: 3, fill: '#b8860b', stroke: '#fff', 'stroke-width': 1.5, opacity: 0 });
    var surface = svgEl('rect', {
      class: 'hover-surface', x: margin.left, y: margin.top,
      width: innerWidth, height: innerHeight
    });
    [crosshair, dotSeasonal, dotActual, dotPred, surface].forEach(function (node) { svg.appendChild(node); });

    function place(dot, x, value) {
      if (!isNum(value)) { dot.setAttribute('opacity', 0); return; }
      dot.setAttribute('cx', x.toFixed(2));
      dot.setAttribute('cy', Y(value).toFixed(2));
      dot.setAttribute('opacity', 1);
    }

    function clearHover() {
      crosshair.setAttribute('opacity', 0);
      [dotActual, dotPred, dotSeasonal].forEach(function (dot) { dot.setAttribute('opacity', 0); });
      if (readout) readout.textContent = 'Point at the chart to read individual hours.';
    }

    surface.addEventListener('pointermove', function (event) {
      var box = svg.getBoundingClientRect();
      if (!box.width) return;
      var px = (event.clientX - box.left) * (width / box.width);
      var time = t0 + ((px - margin.left) / innerWidth) * span;
      var point = points[Math.max(0, Math.min(points.length - 1, nearestIndex(points, time)))];
      var x = X(point.t);
      crosshair.setAttribute('x1', x.toFixed(2));
      crosshair.setAttribute('x2', x.toFixed(2));
      crosshair.setAttribute('opacity', 1);
      place(dotActual, x, point.actual);
      place(dotPred, x, point.prediction);
      if (state.showSeasonal) place(dotSeasonal, x, point.seasonal);
      else dotSeasonal.setAttribute('opacity', 0);
      if (readout) readout.textContent = describePoint(point);
    });
    surface.addEventListener('pointerleave', clearHover);

    host.appendChild(svg);
    if (readout) readout.textContent = 'Point at the chart to read individual hours.';
    setText('chart-summary', summarize(points, rangeLabel));
  }

  function describePoint(point) {
    var parts = [DT_LONG.format(new Date(point.t))];
    parts.push('actual ' + (isNum(point.actual) ? NF0.format(point.actual) : 'n/a'));
    parts.push('forecast ' + (isNum(point.prediction) ? NF0.format(point.prediction) : 'n/a'));
    if (isNum(point.lower) && isNum(point.upper)) {
      parts.push('interval ' + NF0.format(point.lower) + '–' + NF0.format(point.upper));
    }
    if (isNum(point.seasonal)) parts.push('seasonal 168h ' + NF0.format(point.seasonal));
    return parts.join(' · ');
  }

  function summarize(points, rangeLabel) {
    var target = isNum((state.report.intervals || {}).target) ? state.report.intervals.target : 0.9;
    var pairs = 0;
    var absSum = 0;
    var sqSum = 0;
    var actualSum = 0;
    var actualCount = 0;
    var predSum = 0;
    var predCount = 0;
    var covered = 0;
    var coverable = 0;
    var widthSum = 0;
    var widthCount = 0;

    points.forEach(function (point) {
      if (isNum(point.actual)) { actualSum += point.actual; actualCount += 1; }
      if (isNum(point.prediction)) { predSum += point.prediction; predCount += 1; }
      if (isNum(point.actual) && isNum(point.prediction)) {
        var error = point.actual - point.prediction;
        pairs += 1;
        absSum += Math.abs(error);
        sqSum += error * error;
      }
      if (isNum(point.actual) && isNum(point.lower) && isNum(point.upper)) {
        coverable += 1;
        if (point.actual >= point.lower && point.actual <= point.upper) covered += 1;
        widthSum += point.upper - point.lower;
        widthCount += 1;
      }
    });

    var lines = [];
    lines.push('Chart summary (' + rangeLabel + '): ' + points.length.toLocaleString('en-US') + ' hourly points from '
      + DT_LONG.format(new Date(points[0].t)) + ' to ' + DT_LONG.format(new Date(points[points.length - 1].t)) + '.');
    if (actualCount && predCount) {
      lines.push('Mean actual ' + NF0.format(actualSum / actualCount) + ' rides, mean forecast '
        + NF0.format(predSum / predCount) + ' rides.');
    }
    if (pairs) {
      lines.push('MAE ' + NF2.format(absSum / pairs) + ', RMSE ' + NF2.format(Math.sqrt(sqSum / pairs))
        + ' over ' + pairs.toLocaleString('en-US') + ' scored hours.');
    }
    if (coverable) {
      lines.push('The ' + pct(target, 0) + ' interval contains ' + pct(covered / coverable, 1) + ' of these actual values'
        + (widthCount ? ' (mean width ' + NF0.format(widthSum / widthCount) + ' rides)' : '') + '.');
    }
    return lines.join(' ');
  }

  /* --------------------------------- lab --------------------------------- */

  function wireLab() {
    var select = $('example-select');
    if (select) {
      select.addEventListener('change', function () {
        state.exampleIndex = Number(select.value) || 0;
        clearLive('Example changed — run the model again to get a new live prediction.');
        renderExample();
      });
    }
    var body = $('feature-table-body');
    if (body) {
      body.addEventListener('input', function () {
        clearLive('');
        updateInputState();
      });
      body.addEventListener('change', function () { updateInputState(); });
    }
    var reset = $('reset-inputs');
    if (reset) {
      reset.addEventListener('click', function () {
        clearLive('');
        renderExample();
      });
    }
    var run = $('run-live');
    if (run) run.addEventListener('click', runLive);
  }

  function currentExample() {
    return state.examples[state.exampleIndex] || null;
  }

  function renderLab() {
    var select = $('example-select');
    if (select) {
      select.replaceChildren();
      if (!state.examples.length) {
        var option = document.createElement('option');
        option.value = '';
        option.textContent = 'No saved examples in the report';
        select.appendChild(option);
        select.disabled = true;
      } else {
        select.disabled = false;
        state.examples.forEach(function (example, index) {
          var item = document.createElement('option');
          item.value = String(index);
          var ms = parseTs(example.timestamp);
          item.textContent = '#' + (index + 1) + ' · ' + (isFinite(ms) ? DT_LONG.format(new Date(ms)) : String(example.timestamp || '—'));
          select.appendChild(item);
        });
        select.value = String(state.exampleIndex);
      }
    }
    renderExample();
    renderLive();
  }

  function renderExample() {
    var example = currentExample();
    var bars = $('saved-bars');
    var facts = $('saved-facts');
    var meta = $('example-meta');
    if (!example) {
      if (bars) bars.replaceChildren(make('p', 'muted small', 'No saved example is available.'));
      if (facts) facts.replaceChildren();
      setText('example-meta', 'The report contains no examples array entries.');
      renderFeatureTable(null);
      updateInputState();
      return;
    }
    var features = example.features || {};
    var ms = parseTs(example.timestamp);
    setText('example-meta', 'Target hour · ' + (isFinite(ms) ? DT_LONG.format(new Date(ms)) : String(example.timestamp || '—'))
      + ' · saved exactly as scored in the test block.');

    if (bars) {
      bars.replaceChildren();
      var actual = isNum(example.actual) ? example.actual : null;
      var prediction = isNum(example.prediction) ? example.prediction : null;
      var lower = isNum(example.lower) ? example.lower : null;
      var upper = isNum(example.upper) ? example.upper : null;
      var scale = Math.max(actual || 0, prediction || 0, upper || 0, 1);
      bars.appendChild(miniBar('Actual demand (recorded)', actual, scale, 'actual'));
      bars.appendChild(miniBar('Saved forecast', prediction, scale, 'prediction'));
      if (lower !== null && upper !== null) {
        bars.appendChild(miniBar('Interval width', upper - lower, scale, 'interval'));
      }
    }

    if (facts) {
      var rows = [
        ['Actual demand', isNum(example.actual) ? NF0.format(example.actual) + ' rides' : '—'],
        ['Saved forecast', isNum(example.prediction) ? NF0.format(example.prediction) + ' rides' : '—'],
        ['Interval', isNum(example.lower) && isNum(example.upper)
          ? NF0.format(example.lower) + ' – ' + NF0.format(example.upper) : '—'],
        ['Width', isNum(example.lower) && isNum(example.upper) ? NF0.format(example.upper - example.lower) + ' rides' : '—'],
        ['Absolute error', isNum(example.actual) && isNum(example.prediction)
          ? NF0.format(Math.abs(example.actual - example.prediction)) + ' rides' : '—']
      ];
      fillFacts(facts, rows);
    }

    renderFeatureTable(example);
    updateInputState();
  }

  function miniBar(label, value, scale, kind) {
    var row = make('div', 'mini-row');
    var head = make('div', 'mini-head');
    head.appendChild(make('span', null, label));
    head.appendChild(make('b', null, isNum(value) ? NF0.format(value) : '—'));
    row.appendChild(head);
    var track = make('div', 'mini-track');
    var fill = make('div', 'mini-fill ' + kind);
    fill.style.width = (isNum(value) ? Math.max(0, Math.min(1, value / scale)) * 100 : 0).toFixed(2) + '%';
    track.appendChild(fill);
    row.appendChild(track);
    return row;
  }

  function renderFeatureTable(example) {
    var body = $('feature-table-body');
    if (!body) return;
    body.replaceChildren();
    if (!example) {
      body.appendChild(emptyRow(4, 'No feature vector available.'));
      return;
    }
    var features = example.features || {};
    FEATURE_META.forEach(function (meta) {
      var row = document.createElement('tr');
      var name = make('th', null, meta.label);
      name.scope = 'row';
      row.appendChild(name);
      row.appendChild(make('td', 'muted small', meta.meaning));

      var valueCell = make('td', 'num');
      if (meta.editable) {
        var input = document.createElement('input');
        input.type = 'number';
        input.id = 'f-' + meta.key;
        input.value = isNum(features[meta.key]) ? String(features[meta.key]) : '';
        input.setAttribute('aria-label', meta.label);
        if (meta.min !== undefined) input.min = String(meta.min);
        if (meta.max !== undefined) input.max = String(meta.max);
        input.step = String(meta.step || 1);
        input.inputMode = 'decimal';
        valueCell.appendChild(input);
      } else {
        valueCell.appendChild(make('span', 'mono', fmt(features[meta.key], meta.digits)));
      }
      row.appendChild(valueCell);
      row.appendChild(make('td', null, meta.editable ? 'Editable — hypothetical' : 'From saved example'));
      body.appendChild(row);
    });
  }

  function collectFeatures() {
    var example = currentExample();
    if (!example) return { ok: false, message: 'No saved example is available to start from.' };
    var features = Object.assign({}, example.features || {});
    var dirty = false;
    for (var i = 0; i < FEATURE_META.length; i += 1) {
      var meta = FEATURE_META[i];
      if (!meta.editable) continue;
      var input = $('f-' + meta.key);
      if (!input) continue;
      var raw = String(input.value).trim();
      if (!raw) return { ok: false, message: 'Enter a value for “' + meta.label + '”.' };
      var value = Number(raw);
      if (!isFinite(value)) return { ok: false, message: '“' + meta.label + '” must be a number.' };
      if (meta.step === 1 && Math.round(value) !== value) {
        return { ok: false, message: '“' + meta.label + '” must be a whole number of rides.' };
      }
      if (meta.min !== undefined && value < meta.min) {
        return { ok: false, message: '“' + meta.label + '” must be at least ' + meta.min + '.' };
      }
      if (meta.max !== undefined && value > meta.max) {
        return { ok: false, message: '“' + meta.label + '” must be at most ' + meta.max + '.' };
      }
      features[meta.key] = value;
      var base = Number((example.features || {})[meta.key]);
      if (!isFinite(base) || Math.abs(base - value) > 1e-9) dirty = true;
    }
    return { ok: true, features: features, dirty: dirty };
  }

  function updateInputState() {
    var result = collectFeatures();
    var errorNode = $('input-error');
    var whatIf = $('whatif-note');
    var run = $('run-live');
    var hasExample = !!currentExample();

    if (!result.ok && hasExample) {
      if (errorNode) {
        errorNode.textContent = result.message;
        show(errorNode, true);
      }
    } else {
      show(errorNode, false);
    }
    if (whatIf) show(whatIf, !!(result.ok && result.dirty));
    if (run) run.disabled = !(result.ok && hasExample && state.health === 'ready' && !state.busy);
    return result;
  }

  function clearLive(message) {
    state.liveEpoch += 1;
    state.live = null;
    state.liveError = '';
    state.liveNotice = message || '';
    renderLive();
  }

  /* ------------------------------ live model ----------------------------- */

  function probeHealth() {
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    var timer = window.setTimeout(function () { if (controller) controller.abort(); }, 4000);
    var options = { cache: 'no-store', headers: { Accept: 'application/json' } };
    if (controller) options.signal = controller.signal;

    fetch(HEALTH_URL, options).then(function (response) {
      if (!response.ok) {
        setHealth('absent', 'The health check returned HTTP ' + response.status + ' — this page is probably served as static files.');
        return null;
      }
      return response.text();
    }).then(function (text) {
      if (text === null) return;
      var data = null;
      try { data = JSON.parse(text); } catch (err) { data = null; }
      if (!data || typeof data !== 'object') {
        setHealth('absent', 'The health endpoint answered with non-JSON content, which is typical for a static file server.');
        return;
      }
      if (data.ready === true) setHealth('ready', '');
      else setHealth('not-ready', 'The service answered but reports ready: false.');
    }).catch(function (err) {
      var reason = err && err.name === 'AbortError' ? 'The health check timed out.' : 'No response from a local API.';
      setHealth('absent', reason);
    }).then(function () {
      window.clearTimeout(timer);
    });
  }

  function setHealth(mode, detail) {
    state.health = mode;
    state.healthDetail = detail || '';
    var chip = $('live-chip');
    var title = $('live-mode-title');
    var body = $('live-mode-body');
    var note = $('live-note');

    if (mode === 'ready') {
      if (chip) { chip.className = 'chip chip-ok'; chip.textContent = 'Live API ready'; }
      if (title) title.textContent = 'Live predictions enabled';
      if (body) body.textContent = 'The trained model is available on this computer. Choose an example or edit its inputs to make a new prediction.';
      if (note) note.textContent = 'The model, interval and version below come from the local service. Nothing is sent anywhere else.';
    } else if (mode === 'not-ready') {
      if (chip) { chip.className = 'chip chip-warn'; chip.textContent = 'API not ready'; }
      if (title) title.textContent = 'Local API reachable but not ready';
      if (body) body.textContent = 'The dashboard is running, but no trained model is loaded. Follow the local setup guide to enable new predictions.';
      if (note) note.textContent = 'Train the local model and restart the service to enable new predictions.';
    } else if (mode === 'absent') {
      if (chip) { chip.className = 'chip chip-neutral'; chip.textContent = 'Saved replay'; }
      if (title) title.textContent = 'Saved replay mode';
      if (body) body.textContent = 'Explore the saved historical results here. Download and run the project locally to make new predictions.';
      if (note) note.textContent = 'Without a local API the lab shows only the historical example recorded in the report.';
    } else {
      if (chip) { chip.className = 'chip chip-neutral'; chip.textContent = 'Checking API…'; }
      if (title) title.textContent = 'Checking the local API…';
      if (body) body.textContent = 'Saved replay · Start the local API to run new predictions';
      if (note) note.textContent = 'Checking the local API for a live model.';
    }
    updateInputState();
    renderLive();
  }

  function setBusy(busy) {
    state.busy = busy;
    var run = $('run-live');
    if (run) {
      run.textContent = busy ? 'Running…' : 'Run live prediction';
      run.disabled = busy || state.health !== 'ready';
    }
    if (!busy) updateInputState();
  }

  function runLive() {
    var result = updateInputState();
    if (!result.ok || state.health !== 'ready' || state.busy) return;
    state.live = null;
    state.liveError = '';
    state.liveNotice = '';
    renderLive();
    setBusy(true);
    var epoch = ++state.liveEpoch;
    var controller = new AbortController();
    var timeout = window.setTimeout(function () { controller.abort(); }, 15000);

    fetch(PREDICT_URL, {
      method: 'POST',
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ features: result.features })
    }).then(function (response) {
      return response.text().then(function (text) {
        var data = null;
        try { data = JSON.parse(text); } catch (err) { data = null; }
        if (!response.ok) {
          var detail = data && data.detail ? data.detail : null;
          var suffix = detail ? ' · ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)) : '';
          throw new Error('HTTP ' + response.status + suffix);
        }
        return data;
      });
    }).then(function (data) {
      if (epoch !== state.liveEpoch) return;
      if (!data || typeof data !== 'object') throw new Error('the response body was not JSON');
      if (!isNum(data.prediction)) throw new Error('the response did not include a numeric prediction');
      state.live = {
        prediction: data.prediction,
        lower: isNum(data.lower) ? data.lower : null,
        upper: isNum(data.upper) ? data.upper : null,
        modelVersion: data.model_version === undefined ? null : String(data.model_version),
        predictionId: data.prediction_id === undefined ? null : String(data.prediction_id),
        latency: isNum(data.latency_ms) ? data.latency_ms : null,
        hypothetical: result.dirty
      };
    }).catch(function (err) {
      if (epoch !== state.liveEpoch) return;
      state.liveError = err && err.message ? err.message : 'unknown error';
    }).then(function () {
      window.clearTimeout(timeout);
      setBusy(false);
      renderLive();
    });
  }

  function renderLive() {
    var host = $('live-result');
    var errorNode = $('live-error');
    if (!host) return;
    host.replaceChildren();

    if (errorNode) {
      if (state.liveError) {
        errorNode.textContent = 'Live prediction failed: ' + state.liveError;
        show(errorNode, true);
      } else {
        show(errorNode, false);
      }
    }

    if (state.live) {
      var hero = make('div', 'live-hero');
      hero.appendChild(make('p', 'stat-label', state.live.hypothetical ? 'Hypothetical live prediction' : 'Live prediction'));
      hero.appendChild(make('p', 'live-value', NF0.format(state.live.prediction) + ' rides'));
      if (state.live.lower !== null && state.live.upper !== null) {
        hero.appendChild(make('p', 'live-band', '90% interval ' + NF0.format(state.live.lower) + ' – ' + NF0.format(state.live.upper)));
      }
      host.appendChild(hero);

      var meta = make('div', 'live-meta');
      if (state.live.modelVersion !== null) meta.appendChild(make('span', 'chip chip-neutral', 'model ' + state.live.modelVersion));
      if (state.live.latency !== null) meta.appendChild(make('span', 'chip chip-neutral', 'server latency ' + NF1.format(state.live.latency) + ' ms'));
      if (state.live.predictionId !== null) meta.appendChild(make('span', 'chip chip-neutral', 'id ' + shorten(state.live.predictionId, 22)));
      host.appendChild(meta);

      var note = state.live.hypothetical
        ? 'These inputs were edited, so this is a what-if scenario. The model encodes correlation, not cause and effect, and the value is deliberately not compared with the recorded actual demand.'
        : 'Submitted exactly as the saved example features. The output is still a model estimate, not an observation.';
      host.appendChild(make('p', 'muted small', note));
      return;
    }

    var message = state.liveNotice || (state.health === 'ready'
      ? 'No live prediction yet. Run the model to score the feature vector above.'
      : 'Saved replay · Start the local API to run new predictions.');
    host.appendChild(make('p', 'live-empty', message));
  }

  /* ------------------------------ monitoring ----------------------------- */

  function wireMonitoring() {
    var radios = document.querySelectorAll('input[name="monitor-window"]');
    Array.prototype.forEach.call(radios, function (radio) {
      radio.addEventListener('change', function () {
        if (!radio.checked) return;
        state.monitorWindow = radio.value;
        if (state.report) renderMonitoring();
      });
    });
  }

  function statusLabel(status) {
    if (status === 'stable') return 'Stable';
    if (status === 'review') return 'Review';
    if (status === 'insufficient_data') return 'Insufficient data';
    return String(status || 'unknown');
  }

  function statusNote(status, windowName) {
    if (status === 'stable') {
      return 'No feature in the ' + windowName + ' crosses the PSI threshold: the distribution is close to the calibration reference.';
    }
    if (status === 'review') {
      return 'At least one feature in the ' + windowName + ' crosses the PSI threshold. Treat it as a prompt to investigate the window, not as proof that accuracy dropped.';
    }
    if (status === 'insufficient_data') {
      return 'This window holds fewer than 100 hours, so PSI is not treated as a stable signal.';
    }
    return 'The report did not include a recognised status for this window.';
  }

  function renderMonitoring() {
    var report = state.report;
    var monitoring = report.monitoring || {};
    var observed = monitoring.observed || { features: [] };
    var shifted = monitoring.shifted || { features: [] };
    var current = state.monitorWindow === 'shifted' ? shifted : observed;
    var windowName = state.monitorWindow === 'shifted' ? 'synthetic stress window' : 'observed test window';
    var features = Array.isArray(current.features) ? current.features : [];
    var threshold = isNum(current.threshold) ? current.threshold : 0.2;
    var alerts = features.filter(function (row) { return row.alert === true; });
    var reference = (report.splits || {}).calibration || {};

    var chips = $('monitor-chips');
    if (chips) {
      chips.replaceChildren();
      chips.appendChild(make('span', 'chip chip-neutral', state.monitorWindow === 'shifted' ? 'Synthetic stress test' : 'Observed test window'));
      chips.appendChild(make('span', 'chip chip-neutral', 'n = ' + fmt(current.n, 0) + ' hours'));
      chips.appendChild(make('span', 'chip ' + (current.status === 'stable' ? 'chip-ok' : current.status === 'review' ? 'chip-warn' : 'chip-neutral'),
        'State: ' + statusLabel(current.status)));
      chips.appendChild(make('span', 'chip chip-neutral', 'Threshold ' + NF2.format(threshold)));
      chips.appendChild(make('span', 'chip chip-neutral', alerts.length + ' of ' + features.length + ' features flagged'));
    }
    setText('monitor-state-note', statusNote(current.status, windowName)
      + ' Reference window: calibration block (' + fmt(reference.n, 0) + ' eligible hours).');

    /* PSI bars */
    var host = $('psi-chart');
    if (host) {
      host.replaceChildren();
      if (!features.length) {
        host.appendChild(make('p', 'muted small', 'No PSI values were reported for this window.'));
      } else {
        var peak = features.reduce(function (acc, row) {
          return Math.max(acc, isNum(row.psi) ? row.psi : 0);
        }, 0);
        var scale = Math.max(peak, threshold * 1.35, 0.05);
        features.forEach(function (row) {
          host.appendChild(barRow({
            label: String(row.feature),
            ratio: (isNum(row.psi) ? row.psi : 0) / scale,
            thresholdRatio: threshold / scale,
            value: fmt(row.psi, 3) + (row.alert ? ' · alert' : ''),
            alert: row.alert === true
          }));
        });
        setText('psi-sub', 'Reference: calibration window · threshold ' + NF2.format(threshold)
          + ' (dashed) · ' + features.length + ' features checked.');
      }
    }

    /* PSI table */
    var body = $('psi-table-body');
    if (body) {
      body.replaceChildren();
      if (!features.length) body.appendChild(emptyRow(4, 'No PSI values were reported for this window.'));
      features.forEach(function (row) {
        var tr = document.createElement('tr');
        if (row.alert === true) tr.className = 'is-alert';
        var name = make('th', null, String(row.feature));
        name.scope = 'row';
        tr.appendChild(name);
        tr.appendChild(numCell(fmt(row.psi, 4)));
        tr.appendChild(numCell(NF2.format(threshold)));
        var stateCell = make('td');
        stateCell.appendChild(make('span', 'chip ' + (row.alert === true ? 'chip-warn' : 'chip-ok'),
          row.alert === true ? 'Above threshold' : 'Within threshold'));
        tr.appendChild(stateCell);
        body.appendChild(tr);
      });
      setText('psi-table-sub', 'Sorted by PSI, descending · ' + (state.monitorWindow === 'shifted'
        ? 'synthetic stress window' : 'observed test window') + ' · n = ' + fmt(current.n, 0) + ' hours.');
    }

    /* comparison */
    var compare = $('monitor-compare');
    if (compare) {
      fillFacts(compare, [
        ['Observed alerts', alertSummary(observed)],
        ['Shifted alerts', alertSummary(shifted)],
        ['Observed peak PSI', peakPsi(observed)],
        ['Shifted peak PSI', peakPsi(shifted)],
        ['Reference window', 'Calibration block · ' + fmt(reference.n, 0) + ' hours']
      ]);
    }

    /* stress-test explanation is only meaningful for the shifted window */
    setText('psi-note', state.monitorWindow === 'shifted'
      ? 'Synthetic window: prior temperature is raised by 0.25 and clipped to 0–1; demand lags and rolling means are multiplied by 1.6. Some winter features move closer to the autumn reference. Alert count need not increase, and PSI does not measure lost accuracy.'
      : 'PSI is a heuristic distribution check against the calibration window, not a statistical test and not proof that accuracy has dropped. Trend and calendar features can shift between windows by construction.');
  }

  function alertSummary(block) {
    var features = Array.isArray(block && block.features) ? block.features : [];
    var count = features.filter(function (row) { return row.alert === true; }).length;
    return count + ' of ' + features.length + ' features · n = ' + fmt(block && block.n, 0) + ' hours';
  }

  function peakPsi(block) {
    var features = Array.isArray(block && block.features) ? block.features : [];
    var peak = null;
    var name = '—';
    features.forEach(function (row) {
      if (isNum(row.psi) && (peak === null || row.psi > peak)) { peak = row.psi; name = String(row.feature); }
    });
    return peak === null ? '—' : fmt(peak, 4) + ' · ' + name;
  }

  /* -------------------------------- methods ------------------------------ */

  function renderMethods() {
    var report = state.report;
    var data = report.dataset || {};
    var splits = report.splits || {};
    var config = report.configuration && typeof report.configuration === 'object' ? report.configuration : {};
    var benchmark = report.benchmark || {};

    var facts = [
      ['Raw source rows', fmt(data.raw_rows, 0)],
      ['Missing hourly timestamps', fmt(data.missing_hours, 0)],
      ['Eligible rows after features', fmt(data.eligible_rows, 0)],
      ['Coverage period', String(data.start || '—').slice(0, 19).replace('T', ' ') + ' → ' + String(data.end || '—').slice(0, 19).replace('T', ' ')],
      ['CSV SHA-256', shorten(data.csv_sha256 || '—', 22), String(data.csv_sha256 || '')],
      ['Selected model', String(report.selected_model || '—')],
      ['Run id', shorten(report.run_id, 26), String(report.run_id || '')],
      ['Created at', String(report.created_at || '—')]
    ];
    if (isNum(report.training_seconds)) facts.push(['Training time on CPU', NF1.format(report.training_seconds) + ' s']);
    if (typeof report.artifact_sha256 === 'string') facts.push(['Model artifact SHA-256', shorten(report.artifact_sha256, 22), report.artifact_sha256]);
    if (typeof benchmark.scope === 'string') facts.push(['Latency scope', benchmark.scope]);
    if (typeof benchmark.platform === 'string') facts.push(['Benchmark host', benchmark.platform]);
    if (typeof benchmark.processor === 'string') facts.push(['Processor', benchmark.processor]);
    fillFacts($('dataset-facts'), facts);
    var hashCell = $('dataset-facts');
    if (hashCell) {
      var dds = hashCell.querySelectorAll('dd');
      if (dds[4]) dds[4].title = String(data.csv_sha256 || '');
    }

    /* optional fitted configuration, when the run recorded it */
    var configRows = [];
    if (config.parameters && typeof config.parameters === 'object') {
      Object.keys(config.parameters).forEach(function (key) {
        configRows.push([String(key).replace(/_/g, ' '), String(config.parameters[key])]);
      });
    }
    if (config.seed !== undefined) configRows.push(['Random seed', String(config.seed)]);
    if (typeof config.sklearn_version === 'string') configRows.push(['scikit-learn version', config.sklearn_version]);
    if (typeof config.split_protocol === 'string') configRows.push(['Split protocol', config.split_protocol]);
    if (typeof config.feature_protocol === 'string') configRows.push(['Feature protocol', config.feature_protocol]);
    if (Array.isArray(config.features)) configRows.push(['Features used', String(config.features.length)]);
    if (typeof config.data_sha256 === 'string') configRows.push(['Feature matrix SHA-256', shorten(config.data_sha256, 22), config.data_sha256]);
    fillFacts($('config-facts'), configRows);
    show($('config-heading'), configRows.length > 0);

    var table = $('splits-table-body');
    if (table) {
      table.replaceChildren();
      ['train', 'validation', 'calibration', 'test'].forEach(function (name) {
        var block = splits[name];
        if (!block) return;
        var row = document.createElement('tr');
        if (name === 'test') row.className = 'is-selected';
        var head = make('th', null, name.charAt(0).toUpperCase() + name.slice(1));
        head.scope = 'row';
        row.appendChild(head);
        row.appendChild(numCell(fmt(block.n, 0)));
        row.appendChild(make('td', 'mono small', String(block.start || '—').slice(0, 16).replace('T', ' ')));
        row.appendChild(make('td', 'mono small', String(block.end || '—').slice(0, 16).replace('T', ' ')));
        row.appendChild(make('td', 'small', SPLIT_USE[name] || ''));
        table.appendChild(row);
      });
    }

    var validationBody = $('validation-table-body');
    var validation = Array.isArray(report.validation) ? report.validation.slice() : [];
    if (validationBody) {
      validationBody.replaceChildren();
      if (!validation.length) validationBody.appendChild(emptyRow(4, 'No validation metrics were reported.'));
      validation.sort(function (a, b) { return (a.mae || 0) - (b.mae || 0); });
      validation.forEach(function (row, index) {
        var tr = document.createElement('tr');
        if (index === 0) tr.className = 'is-selected';
        var head = make('th', null, String(row.model));
        head.scope = 'row';
        if (row.parameters) head.appendChild(make('div', 'small muted', JSON.stringify(row.parameters)));
        if (index === 0) head.appendChild(make('span', 'row-tag', 'selected'));
        tr.appendChild(head);
        tr.appendChild(numCell(fmt(row.n, 0)));
        tr.appendChild(numCell(fmt(row.mae, 2)));
        tr.appendChild(numCell(fmt(row.rmse, 2)));
        validationBody.appendChild(tr);
      });
    }

    fillCaveats($('methods-caveats'), Array.isArray(report.caveats) ? report.caveats : [], false);
  }

  /* --------------------------------- start -------------------------------- */

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
